# ==============================================================
# VAE + GAN (64x64) — Projeto final alinhado (VAEs & GANs)
# Pedro Furtado Cunha - 837711
# Ígor Almeida Polegato -
# André Fernando Machado - 837864
# ==============================================================

from __future__ import annotations
import os, glob, random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

# ======================
#  Utilidades gerais
# ======================

def set_seed(seed=42):
    """Reprodutibilidade básica."""
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # determinístico e um pouco mais lento
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_grid(x: torch.Tensor, nrow=8) -> torch.Tensor:
    """Clampa para [0,1] e cria um grid para salvar como PNG."""
    return make_grid(x.clamp(0,1), nrow=nrow)


@dataclass
class Args:
    # Dados / execução
    dataset: str = "Flat"                 # MNIST | FashionMNIST | ImageFolder | Flat
    data_dir: str = "./data/forests/images"  # Flat: pasta com imagens | ImageFolder: raiz com subpastas
    out_dir: str = "./outputs"
    batch_size: int = 128
    lr: float = 2e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs_vae: int = 8
    epochs_gan: int = 12
    latent_dim: int = 128
    save_every: int = 1
    num_samples: int = 64
    # Canais e transforms
    in_ch: int = 3             # 1=grayscale (mais leve) | 3=RGB (dataset Kaggle é RGB)
    grayscale: int = 0         # força grayscale quando in_ch==1
    # Estabilidade / opções
    initG_from_vae: int = 0    # 1 copia pesos do decoder do VAE para G no início do GAN
    label_smoothing: float = 1.0  # alvo dos reais p/ D (1.0 = sem smoothing; ex.: 0.9)
    spectral_norm: int = 0     # 1 aplica SpectralNorm no D
    ema: int = 0               # 1 usa EMA no G (melhora amostras)
    ema_decay: float = 0.999
    amp: int = 0               # 1 ativa mixed precision (economiza VRAM, treina mais rápido)


# ======================
#  Datasets (64x64)
# ======================

class FlatFolder(Dataset):
    """
    Lê TODAS as imagens de uma pasta (recursivo), sem exigir subpastas de classe.
    Perfeito para usar a pasta "images" do Kaggle (florestas).
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        for ext in self.exts:
            self.paths += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
        if len(self.paths) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {root}\n"
                               f"Verifique o caminho e a extensão dos arquivos.")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0  # rótulo dummy (não usamos)


def get_loaders(args: Args) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Retorna DataLoader(s) com imagens (64x64) em [0,1].
    - MNIST e FashionMNIST: treino+teste
    - ImageFolder: um DataLoader (treino)
    - Flat (Kaggle florestas): um DataLoader (treino)
    BLINDAGEM DE CANAIS:
      - se in_ch==1 e grayscale==1 -> força 1 canal
      - se in_ch==3             -> força 3 canais (útil para MNIST/Fashion)
    """
    tfms: List[torch.nn.Module] = [T.Resize((64,64))]
    if args.in_ch == 1 and args.grayscale:
        tfms.append(T.Grayscale(num_output_channels=1))
    if args.in_ch == 3:
        tfms.append(T.Grayscale(num_output_channels=3))  # garante 3 canais em MNIST/Fashion
    tfms.append(T.ToTensor())
    transform = T.Compose(tfms)

    if args.dataset == "MNIST":
        tr = tv.datasets.MNIST(args.data_dir, train=True,  transform=transform, download=True)
        te = tv.datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
        return (DataLoader(tr, args.batch_size, True,  num_workers=2, pin_memory=True),
                DataLoader(te, args.batch_size, False, num_workers=2, pin_memory=True))

    if args.dataset == "FashionMNIST":
        tr = tv.datasets.FashionMNIST(args.data_dir, train=True,  transform=transform, download=True)
        te = tv.datasets.FashionMNIST(args.data_dir, train=False, transform=transform, download=True)
        return (DataLoader(tr, args.batch_size, True,  num_workers=2, pin_memory=True),
                DataLoader(te, args.batch_size, False, num_workers=2, pin_memory=True))

    if args.dataset == "ImageFolder":
        ds = tv.datasets.ImageFolder(args.data_dir, transform=transform)
        return (DataLoader(ds, args.batch_size, True, num_workers=2, pin_memory=True), None)

    if args.dataset == "Flat":
        ds = FlatFolder(args.data_dir, transform=transform)
        return (DataLoader(ds, args.batch_size, True, num_workers=2, pin_memory=True), None)

    raise ValueError("dataset deve ser: MNIST | FashionMNIST | ImageFolder | Flat")


# ======================
#  VAE (64→32→16→8 → 8→16→32→64)
# ======================

class Encoder(nn.Module):
    """Encoder conv: 64→32→16→8 (stride=2). Saídas: mu e logvar (vetores latentes)."""
    def __init__(self, in_ch=3, latent_dim=128):
        super().__init__()
        c = 32
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c, 4, 2, 1),   nn.BatchNorm2d(c),   nn.LeakyReLU(0.2, True),  # 64→32
            nn.Conv2d(c, c*2, 4, 2, 1),     nn.BatchNorm2d(c*2), nn.LeakyReLU(0.2, True),  # 32→16
            nn.Conv2d(c*2, c*4, 4, 2, 1),   nn.BatchNorm2d(c*4), nn.LeakyReLU(0.2, True),  # 16→8
        )
        self.flat = c*4*8*8
        self.fc_mu     = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Decoder deconv: 8→16→32→64 (stride=2). Saída: LOGITS (sem sigmoid)."""
    def __init__(self, out_ch=3, latent_dim=128):
        super().__init__()
        c = 32
        self.fc = nn.Linear(latent_dim, c*4*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c*4, c*2, 4, 2, 1), nn.BatchNorm2d(c*2), nn.ReLU(True),   # 8→16
            nn.ConvTranspose2d(c*2, c,   4, 2, 1), nn.BatchNorm2d(c),   nn.ReLU(True),   # 16→32
            nn.ConvTranspose2d(c, out_ch,4, 2, 1),                                         # 32→64 (logits)
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 32*4, 8, 8)
        return self.deconv(h)  # logits


class VAE(nn.Module):
    """VAE completo (mu, logvar, reparametrização; saída em logits)."""
    def __init__(self, in_ch=3, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim)
        self.decoder = Decoder(in_ch, latent_dim)

    @staticmethod
    def reparam(mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar


def vae_loss(x_logits, x, mu, logvar):
    """
    Perda clássica do VAE para dados em [0,1]:
    - Reconstrução: BCEWithLogits (mais estável do que sigmoid + BCE)
    - KL: forma fechada para gaussianas diagonais
    """
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum') / x.size(0)
    kld   = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kld, recon.detach(), kld.detach()


# ======================
#  GAN (G = decoder-like | D conv espelhado)
# ======================

class Generator(nn.Module):
    """Gerador = mesmo esqueleto do decoder: de z → imagem 64x64 (logits)."""
    def __init__(self, z_dim=128, out_ch=3):
        super().__init__()
        self.net = Decoder(out_ch, z_dim)
    def forward(self, z):
        return self.net(z)  # logits


class Discriminator(nn.Module):
    """
    Discriminador conv: 64→32→16→8, depois conv 8x8 → logit escalar.
    Spectral Norm opcional para estabilidade.
    """
    def __init__(self, in_ch=3, spectral_norm=False):
        super().__init__()
        c = 32
        def block(cin, cout, use_bn=True):
            conv = nn.Conv2d(cin, cout, 4, 2, 1)
            if spectral_norm: conv = SN(conv)
            layers = [conv]
            if use_bn: layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.main = nn.Sequential(
            block(in_ch, c, use_bn=False),  # 64→32
            block(c, c*2),                  # 32→16
            block(c*2, c*4),                # 16→8
        )
        last = nn.Conv2d(c*4, 1, 8)        # 8→1 (logit)
        self.last = SN(last) if spectral_norm else last

    def forward(self, x):
        h = self.main(x)
        return self.last(h).view(x.size(0))  # logits (B,)


def bce_logits(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


class EMA:
    """Exponential Moving Average dos pesos do Gerador (opcional)."""
    def __init__(self, model: nn.Module, decay=0.999):
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k,v in model.state_dict().items():
            self.shadow[k].mul_((self.decay)).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ======================
#  Treinos (VAE e GAN)
# ======================

def train_vae(args: Args, train: DataLoader, test: Optional[DataLoader], outdir: str) -> VAE:
    dev = args.device
    vae = VAE(in_ch=args.in_ch, latent_dim=args.latent_dim).to(dev)
    opt = optim.Adam(vae.parameters(), lr=args.lr)

    ensure_dir(outdir)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    for ep in range(1, args.epochs_vae+1):
        vae.train()
        pbar = tqdm(train, desc=f"[VAE] {ep}/{args.epochs_vae}")
        for x, _ in pbar:
            x = x.to(dev)

            # ----- PATCH DE CANAIS (blindagem 1↔3) -----
            if args.in_ch == 1 and x.size(1) == 3:  # modelo 1 canal, dataset RGB
                x = x[:, :1]
            if args.in_ch == 3 and x.size(1) == 1:  # modelo RGB, dataset 1 canal (MNIST/Fashion)
                x = x.repeat(1, 3, 1, 1)
            # ------------------------------------------

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                logits, mu, logvar = vae(x)
                loss, recon, kld = vae_loss(logits, x, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}", rec=f"{recon.item():.3f}", kl=f"{kld.item():.3f}")

        if ep % args.save_every == 0:
            vae.eval()
            with torch.no_grad():
                # Reconstruções
                x,_ = next(iter(train))
                x = x.to(dev)
                if args.in_ch == 1 and x.size(1) == 3: x = x[:, :1]
                if args.in_ch == 3 and x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
                rec = torch.sigmoid(vae(x)[0])
                save_image(to_grid(torch.cat([x[:32], rec[:32]],0), 8),
                           os.path.join(outdir,f"vae_recon_e{ep}.png"))

                # Amostras do prior (z ~ N(0,1))
                z = torch.randn(args.num_samples, args.latent_dim, device=dev)
                samp = torch.sigmoid(vae.decoder(z))
                save_image(to_grid(samp,8), os.path.join(outdir,f"vae_samples_e{ep}.png"))

    torch.save(vae.state_dict(), os.path.join(outdir,"vae.pt"))
    return vae


def train_gan(args: Args, train: DataLoader, outdir: str, z_dim: int, vae_decoder: Optional[Decoder]):
    dev = args.device
    G = Generator(z_dim, out_ch=args.in_ch).to(dev)
    D = Discriminator(in_ch=args.in_ch, spectral_norm=bool(args.spectral_norm)).to(dev)

    # Inicializar G com decoder do VAE (opcional)
    if args.initG_from_vae and vae_decoder is not None:
        G.net.load_state_dict(vae_decoder.state_dict())

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))

    ema = EMA(G, args.ema_decay) if args.ema else None
    scalerG = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    scalerD = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    ensure_dir(outdir)

    for ep in range(1, args.epochs_gan+1):
        pbar = tqdm(train, desc=f"[GAN] {ep}/{args.epochs_gan}")
        for x,_ in pbar:
            x = x.to(dev)

            # ----- PATCH DE CANAIS (blindagem 1↔3) -----
            if args.in_ch == 1 and x.size(1) == 3:  # modelo 1 canal, dataset RGB
                x = x[:, :1]
            if args.in_ch == 3 and x.size(1) == 1:  # modelo RGB, dataset 1 canal
                x = x.repeat(1, 3, 1, 1)
            # ------------------------------------------

            b = x.size(0)

            # 1) Atualiza D: reais=1 (ou 0.9 se smoothing), fakes=0
            z = torch.randn(b, z_dim, device=dev)
            with torch.no_grad():
                fake_logits = G(z)
                x_fake = torch.sigmoid(fake_logits)   # imagem [0,1]
            optD.zero_grad(set_to_none=True)
            y_real = torch.full((b,), fill_value=args.label_smoothing, device=dev, dtype=torch.float32)
            y_fake = torch.zeros(b, device=dev, dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                Dr = D(x)
                Df = D(x_fake)
                lossD = bce_logits(Dr, y_real) + bce_logits(Df, y_fake)
            scalerD.scale(lossD).backward()
            scalerD.step(optD); scalerD.update()

            # 2) Atualiza G (non-saturating): faz D dizer 1 para imagens geradas
            z = torch.randn(b, z_dim, device=dev)
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                Df = D(torch.sigmoid(G(z)))
                lossG = bce_logits(Df, torch.ones_like(Df, dtype=torch.float32))
            scalerG.scale(lossG).backward()
            scalerG.step(optG); scalerG.update()

            if ema: ema.update(G)

            pbar.set_postfix(lossD=f"{lossD.item():.3f}", lossG=f"{lossG.item():.3f}")

        if ep % args.save_every == 0:
            with torch.no_grad():
                z = torch.randn(args.num_samples, z_dim, device=dev)
                if ema:
                    G_ema = Generator(z_dim, args.in_ch).to(dev)
                    G_ema.load_state_dict(G.state_dict()); ema.copy_to(G_ema)
                    samp = torch.sigmoid(G_ema(z))
                else:
                    samp = torch.sigmoid(G(z))
                save_image(to_grid(samp,8), os.path.join(outdir, f"gan_samples_e{ep}.png"))

    torch.save(G.state_dict(), os.path.join(outdir,"gan_G.pt"))
    torch.save(D.state_dict(), os.path.join(outdir,"gan_D.pt"))


# ======================
#  Main / CLI
# ======================

def quick_tests(in_ch=3):
    """Testes de forma (sanidade) — roda em CPU/GPU rapidinho antes do treino real."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(in_ch, 64).to(dev)
    x = torch.randn(4, in_ch, 64,64, device=dev)
    logits, mu, logvar = vae(x)
    assert logits.shape==(4,in_ch,64,64) and mu.shape==(4,64) and logvar.shape==(4,64)
    G = Generator(64, in_ch).to(dev); D = Discriminator(in_ch).to(dev)
    z = torch.randn(4,64, device=dev)
    out = D(torch.sigmoid(G(z)))
    assert out.shape==(4,), "Discriminador deve retornar 1 logit por imagem"


def parse_args()->Args:
    import argparse
    p = argparse.ArgumentParser(description="VAE + GAN (64x64) — Florestas (Kaggle) e datasets didáticos")
    # Dados/execução
    p.add_argument('--dataset', type=str, default='Flat',
                   choices=['MNIST','FashionMNIST','ImageFolder','Flat'])
    p.add_argument('--data_dir', type=str, default='./data/forests/images')
    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--epochs_vae', type=int, default=8)
    p.add_argument('--epochs_gan', type=int, default=12)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=1)
    p.add_argument('--num_samples', type=int, default=64)
    # Canais / transforms
    p.add_argument('--in_ch', type=int, default=3)
    p.add_argument('--grayscale', type=int, default=0)
    # Estabilidade / opções
    p.add_argument('--initG_from_vae', type=int, default=0)
    p.add_argument('--label_smoothing', type=float, default=1.0)
    p.add_argument('--spectral_norm', type=int, default=0)
    p.add_argument('--ema', type=int, default=0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--amp', type=int, default=0)
    a = p.parse_args()
    set_seed(a.seed)
    return Args(dataset=a.dataset, data_dir=a.data_dir, out_dir=a.out_dir,
                batch_size=a.batch_size, lr=a.lr, epochs_vae=a.epochs_vae,
                epochs_gan=a.epochs_gan, latent_dim=a.latent_dim,
                save_every=a.save_every, num_samples=a.num_samples,
                in_ch=a.in_ch, grayscale=a.grayscale,
                initG_from_vae=a.initG_from_vae, label_smoothing=a.label_smoothing,
                spectral_norm=a.spectral_norm, ema=a.ema, ema_decay=a.ema_decay,
                amp=a.amp)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    quick_tests(args.in_ch)

    # Carregamento de dados
    train, test = get_loaders(args)

    # ===== Treino do VAE =====
    vae_out = os.path.join(args.out_dir, "vae"); ensure_dir(vae_out)
    vae = train_vae(args, train, test, vae_out)

    # ===== Treino do GAN =====
    gan_out = os.path.join(args.out_dir, "gan"); ensure_dir(gan_out)
    vae_dec = vae.decoder if args.initG_from_vae else None
    train_gan(args, train, gan_out, z_dim=args.latent_dim, vae_decoder=vae_dec)


if __name__ == "__main__":
    main()
