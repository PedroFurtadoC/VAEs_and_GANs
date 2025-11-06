# VAE + GAN (64×64) — Florestas (Kaggle) e Datasets Didáticos

Implementação didática em **PyTorch** de:

* **VAE**: `64 → 32 → 16 → 8 → 16 → 32 → 64` (saída em **logits**, perda = **BCEWithLogits + KL**).
* **GAN**: **Gerador** = *decoder-like* (logits) e **Discriminador** conv espelhado (64→…→8→**logit escalar**).

Suporta **MNIST / FashionMNIST / ImageFolder / Flat** (pasta “solta” com imagens – ideal para o Kaggle de florestas).

---

## Requisitos

* Python **3.12** (recomendado)
* PyTorch **2.5+** (GPU opcional)
* Pacotes: `torch torchvision torchaudio numpy tqdm pillow`

Instalação (escolha UMA das linhas do PyTorch):

```bash
# GPU CUDA 12.1 (NVIDIA)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OU: CPU (sem GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# comuns
python -m pip install numpy tqdm pillow
```

---

## Arquivo principal

* `vae_gan_64.py` (gera saídas em `outputs/vae/` e `outputs/gan/`)

---

## Como rodar

### 1) MNIST (didático, 1 canal)

```bash
python vae_gan_64.py --dataset MNIST --in_ch 1 --grayscale 1 \
  --epochs_vae 5 --epochs_gan 5 --batch_size 128
```

### 2) FashionMNIST (didático, 1 canal)

```bash
python vae_gan_64.py --dataset FashionMNIST --in_ch 1 --grayscale 1 \
  --epochs_vae 5 --epochs_gan 5
```

### 3) Kaggle — **Florestas** (modo **Flat**, imagens RGB)

Baixe o ZIP do Kaggle **Augmented Forest Segmentation** e aponte o `--data_dir` para a **pasta `images/`** (dentro do ZIP extraído).

Estrutura típica após extrair:

```
archive/
└── Forest Segmented/
    └── Forest Segmented/
        ├── images/   <-- usar este caminho no --data_dir
        ├── masks/    (ignorar)
        └── meta_data.csv
```

#### Windows (PowerShell) — linha única

```powershell
python vae_gan_64.py --dataset Flat --data_dir "C:\...\Forest Segmented\Forest Segmented\images" `
  --in_ch 3 --grayscale 0 --epochs_vae 8 --epochs_gan 12 --batch_size 64 `
  --label_smoothing 0.9 --spectral_norm 1 --ema 1 --initG_from_vae 1 `
  --out_dir outputs_forest_rgb
```

> **Dica:** mantenha as **aspas** no `--data_dir` (há espaços no caminho).
> Em PowerShell, prefira **linha única** para evitar quebras de continuação.

#### Opcional (cinza, mais leve)

```bash
python vae_gan_64.py --dataset Flat --data_dir ./data/forests/images \
  --in_ch 1 --grayscale 1 --epochs_vae 8 --epochs_gan 12 --batch_size 128
```

---

## Saídas (anexe no relatório)

* `outputs/vae/vae_input_e*.png` → **entradas reais** (grid)
* `outputs/vae/vae_recon_e*.png` → **reconstruções** do VAE
* `outputs/vae/vae_samples_e*.png` → **amostras** do VAE (z ~ N(0,1))
* `outputs/gan/gan_samples_e*.png` → **amostras** do GAN por época
* Pesos: `outputs/vae/vae.pt`, `outputs/gan/gan_G.pt`, `outputs/gan/gan_D.pt`

**Como validar rapidamente**

1. Abra `vae_input_e1.png` e `vae_recon_e1.png` → reconstruções devem se parecer com as entradas.
2. Compare `gan_samples_e1.png` vs `gan_samples_e{última}.png` → amostras do GAN melhoram por época.

---

## Flags úteis

* `--in_ch {1|3}`: **1=grayscale**, **3=RGB**
* `--grayscale {0|1}`: quando `in_ch=1`, converte para 1 canal
* `--initG_from_vae 1`: inicia **G** com os pesos do **decoder do VAE**
* `--label_smoothing 0.9`: reais=0.9 no **D** (estabiliza)
* `--spectral_norm 1`: SpectralNorm no **D** (estabiliza)
* `--ema 1`: EMA do **G** (amostras mais estáveis)
* `--amp 1`: mixed precision (GPU: mais rápido/menos VRAM)
* `--out_dir <pasta>`: separa os resultados por experimento

---

## Notas rápidas

* Roda em **CPU** (mais lento) e em **GPU 4 GB** (ex.: RTX 3050) sem sufoco.
* Para **OOM**, reduza `--batch_size` e/ou ative `--amp 1`.
* O dataset de florestas (segmentação) é usado **apenas como banco de imagens** — **as máscaras são ignoradas**.
* Se aparecerem dígitos (MNIST) em vez de florestas, verifique:

  1. `--dataset Flat`
  2. `--data_dir` aponta para a **pasta `images/`**
  3. `--out_dir` diferente (para não confundir com PNGs antigos)
