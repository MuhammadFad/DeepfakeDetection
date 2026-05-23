# Deepfake Detection System

Classify face images as **Real** or **Fake** using three deep learning architectures running in parallel — a CNN, a Vision Transformer, and a Hybrid model. Each prediction comes with a **Grad-CAM heatmap** showing which facial regions triggered the decision, rendered in a self-contained dark-themed HTML report.

Built as an educational project comparing architecture families on binary deepfake classification.

---

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

All models use **ImageNet-pretrained backbones** with fine-tuned 2-class heads. The included checkpoints were trained on the Kaggle 140k Real and Fake Faces dataset.

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
   - Clone the repo
   - Download the [Kaggle 140k dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (you'll need a `kaggle.json` API key)
   - Train all three models (~30 min total on T4)
   - Prompt you to download the three `.pth` checkpoint files
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

### Local training (advanced)

If you have a local NVIDIA GPU, you can also train locally from the project root:

```bash
python scripts/train.py --model xception
python scripts/train.py --model vit_small_patch16_224
python scripts/train.py --model efficientnet_b4

# Resume after a crash — skips models with a valid checkpoint
python scripts/train.py --skip-existing

# Force retrain
python scripts/train.py --model xception --force
```

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
│   ├── dataset.py          ← PyTorch Dataset with augmentation
│   ├── preprocessing.py    ← image loading and normalisation
│   ├── evaluation.py       ← accuracy, confusion matrix, metrics
│   ├── explainability.py   ← Grad-CAM heatmap generation
│   ├── report.py           ← self-contained HTML report builder
│   └── logger.py           ← append-only training log
│
├── scripts/                ← training utilities
│   ├── train.py            ← local fine-tuning pipeline
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

## Limitations

- **Single fake source generalisation.** The included checkpoints were trained on [Kaggle 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (StyleGAN2-generated fakes). Performance will degrade on other fake sources (FaceSwap, Stable Diffusion, DALL-E, Midjourney). For broader generalisation, retrain with a diverse multi-source dataset such as [FaceForensics++](https://github.com/ondyari/FaceForensics) or [DFDC](https://ai.meta.com/datasets/dfdc/).

- **Dataset bias.** Real and fake images often differ in JPEG compression, colour temperature, and resolution beyond facial manipulation. Use the Grad-CAM heatmaps to check whether the model is responding to faces or to background/border artefacts.

- **Binary classification only.** The system outputs Real vs Fake — it does not identify manipulation type (face swap, full synthesis, expression transfer, etc.).
