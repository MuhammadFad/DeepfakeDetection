# Deepfake Detection System

Classify face images as **Real** or **Fake** using three deep learning architectures running in parallel — a CNN, a Vision Transformer, and a Hybrid model. Each prediction comes with a **Grad-CAM heatmap** that shows exactly which facial regions triggered the decision, rendered in a self-contained dark-themed HTML report.

Built as an educational project comparing architecture families on binary deepfake classification.

---

## What It Does

- Runs **XceptionNet**, **ViT-Base/16**, and **EfficientNet-B4** on the same images simultaneously
- Generates a fully self-contained HTML report (no server needed) with:
  - Interactive accuracy comparison bar chart
  - Per-image predictions from all three models
  - **Grad-CAM heatmaps** — red/warm regions = most suspicious, blue/cool = ignored
  - Confusion matrices per model
  - Training loss/accuracy curves (if you fine-tune the models)
- Supports fine-tuning on your own labelled dataset with a two-phase training pipeline

---

## Models

| Model | Architecture | Params | Backbone |
|-------|-------------|--------|----------|
| XceptionNet | CNN | 22M | Depthwise separable convolutions |
| ViT-Base/16 | Vision Transformer | 86M | 16x16 patch self-attention |
| EfficientNet-B4 | Hybrid | 19M | Compound-scaled MobileNet |

All models use **ImageNet-pretrained backbones** with randomly-initialised 2-class heads. Fine-tuning on deepfake data moves accuracy from ~50% (random head) to 85–95%+.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [Git LFS](https://git-lfs.github.com/) — required to download the model checkpoints

```bash
# Install Git LFS (once per machine)
git lfs install
```

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/DeepfakeDetection.git
cd DeepfakeDetection
```

> Git LFS will automatically pull the `.pth` checkpoint files during clone.

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the demo

The repo ships with 16 test images and a pre-trained ViT checkpoint. Just run:

```bash
python main.py
```

The HTML report opens in your browser automatically. That's it.

---

## Getting Training Data

The 16 demo images in `images/test/` are already in the repo — `main.py` works immediately after cloning. To train the models you need a larger dataset.

### Option A — Automatic download (recommended)

Downloads real faces from [randomuser.me](https://randomuser.me) and AI-generated faces from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) and splits them directly into `train/`, `val/`, and `test/`. The target directories exist in the repo via `.gitkeep` placeholders — the downloader fills them in place.

```bash
python download_data.py               # 50 per class (default)
python download_data.py --count 100   # more data = better accuracy
```

Split ratios: **70% train / 15% val / 15% test** (configurable in `config.py`).

| Source | Class | Description |
|--------|-------|-------------|
| randomuser.me | Real | Real profile photos, CC-licensed |
| thispersondoesnotexist.com | Fake | StyleGAN2-generated synthetic faces |

### Option B — Bring your own dataset

Point the splitter at any two directories and it will organise them into the correct layout:

```bash
python prepare_data.py --real /path/to/your/real --fake /path/to/your/fake
```

This copies the images into `images/train/`, `images/val/`, and `images/test/`. Pass `--move` to relocate instead of copy.

---

## Training

After running `download_data.py`:

```bash
# Train all three models (10 epochs total per model)
python train.py

# Train a single model
python train.py --model xception
python train.py --model vit_base_patch16_224
python train.py --model efficientnet_b4

# Override epochs
python train.py --epochs 20 --warmup 5
```

### Two-phase fine-tuning strategy

| Phase | Epochs | What's trained | Learning rate |
|-------|--------|---------------|---------------|
| 1 — Warmup | 3 | Classification head only | 1e-3 |
| 2 — Fine-tune | 7 | Full network (differential LR) | Backbone: 1e-5 / Head: 1e-4 |

Phase 1 prevents the randomly-initialised head from destroying the pretrained backbone's features. Phase 2 uses CosineAnnealingLR and gradient clipping (max norm 1.0).

Best checkpoints are saved to `checkpoints/` and loaded automatically by `main.py`.

> **Memory note:** ViT-Base backpropagation is memory-intensive. Batch size is set to 4 by default (`BATCH_SIZE_TRAIN` in `config.py`). Do not increase it above 8 without a dedicated GPU.

---

## Included Checkpoints

| Model | Checkpoint | Status |
|-------|-----------|--------|
| ViT-Base/16 | `vit_base_patch16_224_best.pth` | Trained (val acc 100% on 7 images) |
| XceptionNet | `xception_best.pth` | Corrupted — retrain with `python train.py --model xception` |
| EfficientNet-B4 | `efficientnet_b4_best.pth` | Corrupted — retrain with `python train.py --model efficientnet_b4` |

Checkpoints are stored with **Git LFS** (the ViT checkpoint is 327 MB). All three are tracked automatically — no manual download needed after cloning.

> The ViT's 100% val accuracy is on a tiny 7-image validation set and likely reflects dataset-specific compression artefacts rather than generalised deepfake detection. The Grad-CAM heatmaps in the report will show you exactly what the model is focusing on.

---

## Project Structure

```
DeepfakeDetection/
│
├── main.py              # Entry point — runs inference and opens the HTML report
├── train.py             # Two-phase fine-tuning pipeline
├── config.py            # All constants, paths, and hyperparameters
├── models.py            # Model loading and inference (auto-loads checkpoints)
├── dataset.py           # PyTorch Dataset with augmentation and split detection
├── preprocessing.py     # Image loading, resizing, and normalisation
├── evaluation.py        # Accuracy, confusion matrix, per-class stats
├── explainability.py    # Grad-CAM heatmap generation (CNN + ViT)
├── report.py            # Self-contained HTML report with Plotly charts
├── download_data.py     # Downloads and splits a sample dataset automatically
├── prepare_data.py      # Splits your own dataset into train/val/test
├── requirements.txt     # Python dependencies
│
├── checkpoints/         # Model weights — tracked with Git LFS
│   ├── vit_base_patch16_224_best.pth   (327 MB, trained)
│   ├── xception_best.pth               (80 MB,  needs retraining)
│   └── efficientnet_b4_best.pth        (68 MB,  needs retraining)
│
└── images/
    ├── test/            # 16 demo images included in the repo
    │   ├── real/        # 8 real faces
    │   └── fake/        # 8 AI-generated faces
    │
    ├── train/           # Populated by download_data.py (gitkeep placeholder only)
    │   ├── real/
    │   └── fake/
    │
    └── val/             # Populated by download_data.py (gitkeep placeholder only)
        ├── real/
        └── fake/
```

> `images/train/` and `images/val/` contain only `.gitkeep` placeholder files — actual images are excluded from the repo. Run `download_data.py` to populate them. The `output/` directory (HTML reports) is also excluded and created automatically by `main.py`.

---

## How Grad-CAM Works

Gradient-weighted Class Activation Mapping (Grad-CAM) computes how much each spatial region in the image contributed to the model's final prediction.

**For CNN models (XceptionNet, EfficientNet):**
The gradient of the predicted class score with respect to the final convolutional feature map is computed. Feature channels are weighted by their global average gradient and summed, producing a coarse spatial map that is upsampled to the input resolution.

**For ViT:**
The last transformer block's layer norm is used as the target. The 196 patch tokens (14x14 grid) are reshaped back into a spatial heatmap before upsampling, giving a spatial view of which patches the model attended to most.

**Reading the heatmaps in the report:**

| Colour | Meaning |
|--------|---------|
| Red / orange | High influence — the model weighted this region heavily |
| Yellow / green | Moderate influence |
| Blue / dark | Low influence — largely ignored |

If the heatmap highlights eye edges, skin boundaries, or hair blending artefacts, the model is detecting genuine manipulation signals. If it highlights backgrounds or image borders, the model may be exploiting dataset-level compression or colour distribution differences rather than actual deepfake artefacts.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12+ |
| RAM | 4 GB | 8 GB |
| GPU | Not required | CUDA GPU (VRAM 6 GB+) |
| Disk | 2 GB | 5 GB (with full training data) |

The system auto-detects CUDA and falls back to CPU. CPU inference takes ~1 second per image per model. Training on CPU takes 20–40 minutes for 10 epochs at batch size 4.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0+ | Model training and inference |
| torchvision | 0.15+ | Image transforms |
| timm | 1.0+ | Pretrained model zoo |
| grad-cam | 1.4+ | Grad-CAM heatmaps |
| opencv-python | 4.7+ | Image loading and processing |
| plotly | 5.14+ | Interactive charts |
| scikit-learn | 1.2+ | Confusion matrix and metrics |
| Pillow | 9.4+ | Image encoding for report |
| tqdm | 4.65+ | Progress bars |
| requests | 2.28+ | Dataset downloader |

---

## Limitations

- **Small training set.** The included checkpoint was trained on 50 images per class. Performance on real-world deepfakes from different sources (FaceSwap, DALL-E, Midjourney) will vary significantly.
- **Dataset bias.** The two data sources (randomuser.me vs thispersondoesnotexist.com) differ in JPEG compression, colour temperature, and background complexity. The model may partially learn these distribution differences rather than facial manipulation signals. The Grad-CAM heatmaps help diagnose this.
- **CPU speed.** ViT-Base/16 is slower than CNNs on CPU due to the attention mechanism's quadratic complexity over patches.
- **Binary classification only.** The system detects Real vs Fake but does not identify the type of manipulation (face swap, expression synthesis, full synthesis, etc.).
