# Deepfake Detection System

Classify face images as **Real** or **Fake** using three deep learning architectures running in parallel — a CNN, a Vision Transformer, and a Hybrid model. Each prediction comes with a **Grad-CAM heatmap** showing which facial regions triggered the decision, rendered in a self-contained dark-themed HTML report.

Built as an educational project comparing architecture families on binary deepfake classification.

---
## Team

| Name | Student ID |
|------|-----------|
| Muhammad Fahad Hussain Rana | FA23-BCS-107 |
| Muhammad Abdullah           | FA23-BCS-091 |
| Muhammad Ahsan Shaikh       | FA23-BCS-099 |

**Course:** Information Security — CUI Lahore   
**Instructor:** Usama Ahmed

## What It Does

- Runs **XceptionNet**, **ViT-Small/16**, and **EfficientNet-B4** on every image in parallel (one thread per model)
- Generates a fully self-contained HTML report (no server needed) with:
  - Interactive accuracy comparison bar chart
  - Per-image predictions from all three models
  - **Grad-CAM heatmaps** — red/warm regions = most suspicious, blue/cool = ignored
  - Confusion matrices per model
  - Training loss/accuracy curves (shown when checkpoints are present)

---

## Models

| Model | Architecture | Params | Backbone |
|-------|-------------|--------|----------|
| XceptionNet | CNN | 22M | Depthwise separable convolutions |
| ViT-Small/16 | Vision Transformer | 22M | 16×16 patch self-attention |
| EfficientNet-B4 | Hybrid | 19M | Compound-scaled MobileNet |

All models use **ImageNet-pretrained backbones** with fine-tuned 2-class heads. The included checkpoints were trained on a 14,000-image subset of the Kaggle 140k Real and Fake Faces dataset (10,000 train / 2,000 val / 2,000 test).

---

## Quick Start

### Prerequisites

- Python 3.10–3.12
- [Git LFS](https://git-lfs.github.com/) — required to pull the model checkpoints

```bash
git lfs install
```

### 1. Clone

```bash
git clone https://github.com/MuhammadFad/DeepfakeDetection.git
cd DeepfakeDetection
```

Git LFS automatically pulls the `.pth` checkpoint files during clone.

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

**CPU (works everywhere):**
```bash
pip install -r requirements.txt
```

**GPU (NVIDIA, recommended for training):**
```bash
pip install -r requirements-gpu.txt
```

### 4. Run inference

```bash
python main.py
```

The HTML report opens in your browser automatically. Place your own images in `images/test/real/` and `images/test/fake/` to test them.

---

## Training

Training is done on **Google Colab** using the included notebook — it's faster, free, and doesn't require a local GPU.

### Steps

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `train_colab.ipynb`
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom — the notebook will:
   - Clone the repo (skipped if already present — reconnect-safe)
   - Download the [Kaggle 140k dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (you'll need a `kaggle.json` API key; skipped if already downloaded)
   - Populate image folders, skipping files that already exist
   - Auto-scale batch size to the T4's available VRAM (typically 96–128)
   - Train all three models (~10 min per model, ~30 min total on T4)
   - Prompt you to download the three `.pth` checkpoint files

**Reconnect-safe:** Every setup cell is idempotent — re-running after a disconnect is always safe. Training cells use `--skip-existing` so finished models are never retrained from scratch.

4. Place the downloaded `.pth` files in `checkpoints/` and commit them with Git LFS:

```bash
git lfs track "*.pth"
git add checkpoints/
git commit -m "Add trained checkpoints"
git push
```

### Training details — two-phase fine-tuning

| Phase | What's trained | Epochs | Learning rate |
|-------|---------------|--------|---------------|
| 1 — Warmup | Classification head only | 4 | 1e-3 |
| 2 — Fine-tune | Full network (differential LR) | up to 16 | Backbone: 1e-5 / Head: 1e-4 |

Phase 1 protects the pretrained backbone from the randomly-initialised head. Phase 2 uses CosineAnnealingLR, gradient clipping (max norm 1.0), and early stopping (patience 5 on val loss). Checkpoints are saved atomically (temp file → rename) so a crash never corrupts a previous best.

**Additional training features:**
- **Automatic Mixed Precision (AMP)** — `torch.amp.autocast` + `GradScaler` for faster GPU training and lower VRAM usage
- **`torch.compile()`** — JIT-compiles the model graph on CUDA for extra throughput (PyTorch 2.0+)
- **Data augmentation** — random horizontal flip, colour jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2), ±10° rotation, and random erasing (p=0.2) applied per-batch
- **Label smoothing** (ε=0.1) — regularises the cross-entropy loss to reduce overconfidence

### Local training (advanced)

If you have a local NVIDIA GPU, you can also train locally from the project root:

```bash
# Train a single model (hardware auto-scales batch size and workers)
python scripts/train.py --model xception --auto-scale
python scripts/train.py --model vit_small_patch16_224 --auto-scale
python scripts/train.py --model efficientnet_b4 --auto-scale

# Train all three models at once
python scripts/train.py --auto-scale

# Resume after a crash — skips models with a valid checkpoint
python scripts/train.py --skip-existing --auto-scale

# Force retrain a specific model
python scripts/train.py --model xception --force

# Override epoch count
python scripts/train.py --epochs 15
```

#### `--auto-scale` (recommended on GPU)

Probes your hardware at startup and maximises throughput:
- Detects available GPU VRAM and snaps batch size to the largest power-of-2 multiple that fits in ~90% of it (128 → 96 → 64 → 32 → 16 → 8)
- Sets CPU workers to `cpu_count − 1` on Linux/macOS, forces 0 on Windows (avoids multiprocessing crashes)

Without `--auto-scale` the training script uses conservative defaults (batch 16, 0 workers).

#### `--cache` (high-VRAM GPUs only)

```bash
python scripts/train.py --model xception --auto-scale --cache
```

Preloads the **entire training split onto GPU VRAM** as FP16 tensors before training starts. GPU-native augmentation (flip, colour jitter, rotation, random erasing) is then applied each epoch entirely on the GPU — eliminating CPU data-loading overhead almost completely.

Only use this if you have enough VRAM to hold the dataset (~12 GB for the full 140k split). The script raises a clear `RuntimeError` if it runs out of memory.

A live log of every epoch is written to `output/training_log.txt`.

### Bring your own dataset

To train on a custom dataset, split it into the project's folder layout first:

```bash
python scripts/prepare_data.py --real path/to/real --fake path/to/fake
```

This copies images into `images/train/`, `images/val/`, and `images/test/` at a 70/15/15 split.

---

## Included Checkpoints

| Model | Checkpoint | Dataset |
|-------|-----------|---------|
| XceptionNet | `xception_best.pth` | Kaggle 140k |
| ViT-Small/16 | `vit_small_patch16_224_best.pth` | Kaggle 140k |
| EfficientNet-B4 | `efficientnet_b4_best.pth` | Kaggle 140k |

Checkpoints are stored with **Git LFS** and pulled automatically on clone.

---

## Project Structure

```
DeepfakeDetection/
│
├── main.py                 ← run inference → opens HTML report
├── train_colab.ipynb       ← Colab training notebook (start here)
├── requirements.txt
├── requirements-gpu.txt
│
├── src/                    ← core library modules
│   ├── config.py           ← all constants, paths, hyperparameters
│   ├── models.py           ← model loading and inference
│   ├── dataset.py          ← PyTorch Dataset + GPU-cached variant with augmentation
│   ├── preprocessing.py    ← image loading and normalisation
│   ├── evaluation.py       ← accuracy, confusion matrix, metrics
│   ├── explainability.py   ← Grad-CAM heatmap generation
│   ├── report.py           ← self-contained HTML report builder
│   └── logger.py           ← append-only training log
│
├── scripts/                ← training utilities
│   ├── train.py            ← local fine-tuning pipeline (AMP, auto-scale, GPU cache)
│   └── prepare_data.py     ← split a custom dataset into train/val/test
│
├── checkpoints/            ← model weights (Git LFS)
│   ├── xception_best.pth
│   ├── vit_small_patch16_224_best.pth
│   └── efficientnet_b4_best.pth
│
└── images/
    ├── test/               ← images evaluated by main.py
    │   ├── real/
    │   └── fake/
    ├── train/              ← populated before training
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

---

## How Grad-CAM Works

Gradient-weighted Class Activation Mapping computes how much each spatial region contributed to the model's prediction.

**For CNN models (XceptionNet, EfficientNet):** The gradient of the predicted class score with respect to the final convolutional feature map is computed. Feature channels are weighted by their global average gradient and summed into a spatial map, then upsampled to input resolution.

**For ViT:** The last transformer block's layer norm is used as the target. The 196 patch tokens (14×14 grid) are reshaped into a spatial heatmap before upsampling.

| Colour | Meaning |
|--------|---------|
| Red / orange | High influence — model weighted this region heavily |
| Yellow / green | Moderate influence |
| Blue / dark | Low influence — largely ignored |

If the heatmap highlights eye edges, skin texture boundaries, or hair blending artefacts, the model is detecting genuine manipulation signals. If it highlights backgrounds or borders, the model may be exploiting dataset-level shortcuts.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| RAM | 4 GB | 8 GB |
| GPU | Not required (inference) | CUDA GPU for training |
| Disk | 1 GB | 5 GB |

CPU inference takes ~1 second per image per model. The system auto-detects CUDA and falls back to CPU.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch / torchvision | Model training and inference |
| timm | Pretrained model zoo (XceptionNet, ViT, EfficientNet) |
| grad-cam | Grad-CAM heatmaps |
| opencv-python | Image loading and processing |
| plotly | Interactive charts in the HTML report |
| scikit-learn | Confusion matrix and metrics |
| Pillow | Image encoding |
| tqdm | Progress bars |

---

## Evaluation Results & Screenshots

This project was evaluated at two distinct scales. Screenshots of both runs are included in the `screenshots/` directory for reference.

### Full Dataset Evaluation (Training Run)

The three model checkpoints were trained and evaluated on a **14,000-image subset** of the [Kaggle 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset, split as follows:

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 10,000 | Gradient updates and weight optimisation |
| Validation | 2,000 | Early stopping and best-checkpoint selection |
| Test | 2,000 | Final held-out accuracy and confusion matrix |

Training was performed on Google Colab (T4 GPU) using `train_colab.ipynb`. The resulting accuracy metrics, confusion matrices, and training curves shown in `screenshots/full_dataset_results/` reflect performance at this scale.

### Demo Evaluation (10-Image Laptop Run)

As required for the course submission, a **live demo** was recorded running `main.py` on a standard student laptop (CPU only, no GPU). Due to hardware constraints, the demo test set was limited to **10 images** (5 real, 5 fake) drawn from `images/test/`. The generated HTML report — including Grad-CAM heatmaps for all three models — is shown in `screenshots/demo_results/`.

> **Note:** The demo's smaller sample size means its per-model accuracy figures and confusion matrices are not statistically representative of the full evaluation. They are included solely to demonstrate the inference pipeline and Grad-CAM visualisation functionality running end-to-end on consumer hardware. For statistically meaningful results, refer to the full dataset evaluation screenshots.

---

## Limitations

- **Single fake source generalisation.** The included checkpoints were trained on [Kaggle 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (StyleGAN2-generated fakes). Performance will degrade on other fake sources (FaceSwap, Stable Diffusion, DALL-E, Midjourney). For broader generalisation, retrain with a diverse multi-source dataset such as [FaceForensics++](https://github.com/ondyari/FaceForensics) or [DFDC](https://ai.meta.com/datasets/dfdc/).

- **Dataset bias.** Real and fake images often differ in JPEG compression, colour temperature, and resolution beyond facial manipulation. Use the Grad-CAM heatmaps to check whether the model is responding to faces or to background/border artefacts.

- **Binary classification only.** The system outputs Real vs Fake — it does not identify manipulation type (face swap, full synthesis, expression transfer, etc.).