# VAE + GAN (64×64) — Florestas (Kaggle) e Datasets Didáticos

Implementação didática em PyTorch de **VAE** (64→32→16→8→16→32→64) e **GAN** (Gerador = decoder-like; Discriminador conv espelhado).
Suporta **MNIST / FashionMNIST / ImageFolder / Flat** (pasta com imagens, ideal pro Kaggle de florestas).

## Requisitos

* Python 3.12 (recomendado)
* PyTorch 2.5+ (GPU opcional)
* Pacotes: `torch torchvision torchaudio numpy tqdm pillow matplotlib`

```bash
# GPU CUDA 12.1 (NVIDIA):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU (sem GPU):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# comuns
python -m pip install numpy tqdm pillow matplotlib
```

## Arquivo principal

* `vae_gan_64.py`

## Como rodar

### 1) MNIST (base didática)

```bash
python vae_gan_64.py --dataset MNIST --epochs_vae 5 --epochs_gan 5 --batch_size 128
```

### 2) FashionMNIST (cinza)

```bash
python vae_gan_64.py --dataset FashionMNIST --epochs_vae 5 --epochs_gan 5
```

### 3) Kaggle — Florestas (pasta “Flat” com imagens RGB)

Baixe o dataset e deixe as imagens em `data/forests/images/` (qualquer hierarquia de subpastas funciona).

```bash
# RGB (3 canais)
python vae_gan_64.py --dataset Flat --data_dir data/forests/images \
  --in_ch 3 --grayscale 0 --epochs_vae 8 --epochs_gan 12 --batch_size 64
```

**Opcional (cinza, mais leve):**

```bash
python vae_gan_64.py --dataset Flat --data_dir data/forests/images \
  --in_ch 1 --grayscale 1 --epochs_vae 8 --epochs_gan 12 --batch_size 128
```

## Saídas (para anexar no relatório)

* `outputs/vae/vae_recon_e*.png` → originais vs reconstruções (VAE)
* `outputs/vae/vae_samples_e*.png` → amostras novas do VAE
* `outputs/gan/gan_samples_e*.png` → amostras do GAN por época
* Pesos: `outputs/vae/vae.pt`, `outputs/gan/gan_G.pt`, `outputs/gan/gan_D.pt`

## Flags úteis

* `--in_ch {1|3}`: 1=grayscale (leve), 3=RGB
* `--grayscale {0|1}`: força converter pra cinza quando `in_ch=1`
* `--initG_from_vae 1`: inicia o **Gerador** com os pesos do **decoder do VAE**
* `--label_smoothing 0.9`: reais=0.9 no Discriminador
* `--spectral_norm 1`: SpectralNorm no Discriminador
* `--ema 1`: EMA do Gerador (amostras mais estáveis)
* `--amp 1`: mixed precision (mais rápido/menos VRAM na GPU)

## Notas rápidas

* Roda na **CPU** (mais lento) e em **GPU 4GB** sem sufoco (ex.: RTX 3050).
* Para erros de memória, **reduza `--batch_size`**.
* O dataset de florestas de segmentação pode ser usado **apenas como banco de imagens** (ignorando máscaras).
