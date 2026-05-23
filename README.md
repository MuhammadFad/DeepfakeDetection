# Deepfake Detection System

Classify face images as **Real** or **Fake** using three deep learning architectures running in parallel — a CNN, a Vision Transformer, and a Hybrid model. Each prediction comes with a **Grad-CAM heatmap** that shows exactly which facial regions triggered the decision, rendered in a self-contained dark-themed HTML report.

Built as an educational project comparing architecture families on binary deepfake classification.

---

## What It Does

- Runs **XceptionNet**, **ViT-Small/16**, and **EfficientNet-B4** on the same images simultaneously
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
| ViT-Small/16 | Vision Transformer | 22M | 16x16 patch self-attention |
| EfficientNet-B4 | Hybrid | 19M | Compound-scaled MobileNet |

ViT-Small is used instead of ViT-Base deliberately — Base (86M params) massively overfits on small datasets. Small (22M) matches the parameter count of the CNN models and generalises far better when data is limited.

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
# Train all three models
python train.py

# Train a single model (recommended — one at a time is easier on your machine)
python train.py --model xception
python train.py --model vit_small_patch16_224
python train.py --model efficientnet_b4

# Resume after a crash — skips models that already have a valid checkpoint
python train.py --skip-existing

# Force retrain even if a checkpoint exists
python train.py --model xception --force

# Override epochs
python train.py --epochs 20 --warmup 5
```

A live log of every epoch is written to `output/training_log.txt` as training runs — useful for monitoring progress or diagnosing crashes.

### Two-phase fine-tuning strategy

| Phase | Epochs | What's trained | Learning rate |
|-------|--------|---------------|---------------|
| 1 — Warmup | 4 | Classification head only | 1e-3 |
| 2 — Fine-tune | up to 16 | Full network (differential LR) | Backbone: 1e-5 / Head: 1e-4 |

Phase 1 prevents the randomly-initialised head from destroying the pretrained backbone's features. Phase 2 uses CosineAnnealingLR, gradient clipping (max norm 1.0), and early stopping (patience 5 on val loss).

Checkpoints are written atomically (temp file → rename) so a crash mid-save never corrupts the previous best checkpoint. Best checkpoints are saved to `checkpoints/` and loaded automatically by `main.py`.

> **Memory note:** Batch size is set to 2 by default (`BATCH_SIZE_TRAIN` in `config.py`). On a machine with 8 GB+ RAM and no GPU this is the safe default. Do not increase above 4 without testing for out-of-memory errors.

---

## Included Checkpoints

| Model | Checkpoint | Status |
|-------|-----------|--------|
| XceptionNet | `xception_best.pth` | Trained (ep 6, val 100%, test **98.7%**) |
| ViT-Small/16 | `vit_small_patch16_224_best.pth` | Trained (ep 15, val 100%, test **100.0%**) |
| EfficientNet-B4 | `efficientnet_b4_best.pth` | Trained (ep 8, val 78%, test **81.6%**) |

Checkpoints are stored with **Git LFS**. All are tracked automatically — no manual download needed after cloning.

> These results were achieved with 200 images per class (280 train / 74 val / 76 test) from the two auto-download sources. See the Limitations section for why high accuracy here does not guarantee generalisation to other fake sources.

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
├── logger.py            # Append-only training log (output/training_log.txt)
├── download_data.py     # Downloads and splits a sample dataset automatically
├── prepare_data.py      # Splits your own dataset into train/val/test
├── requirements.txt     # Python dependencies
│
├── checkpoints/         # Model weights — tracked with Git LFS
│   ├── xception_best.pth               (80 MB,  trained — test 98.7%)
│   ├── vit_small_patch16_224_best.pth  (83 MB,  trained — test 100.0%)
│   └── efficientnet_b4_best.pth        (68 MB,  trained — test 81.6%)
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

The system auto-detects CUDA and falls back to CPU. CPU inference takes ~1 second per image per model. Training one model on CPU takes roughly 10–15 minutes at batch size 2 with early stopping (typically exits well before the 20-epoch cap).

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

- **Single fake source.** `download_data.py` pulls AI-generated faces exclusively from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) (StyleGAN2). Training on only one GAN source means the model partially learns *that generator's fingerprint* rather than general deepfake artefacts. It will underperform on faces from other sources (FaceSwap, DALL-E, Midjourney, Stable Diffusion). For better generalisation, supply a diverse fake set via `prepare_data.py`.

- **Accuracy on the auto-downloader dataset.** With 200 images per class the included checkpoints achieve:

  | Model | Test accuracy |
  |---|---|
  | XceptionNet | 98.7% |
  | ViT-Small/16 | 100.0% |
  | EfficientNet-B4 | 81.6% |

  These numbers look impressive but reflect the limited two-source dataset described above — the task as posed (randomuser.me vs thispersondoesnotexist.com) may be too easy due to differences in JPEG compression and colour statistics. For results that generalise to real-world deepfakes, use `prepare_data.py` with a diverse, multi-source dataset (e.g. [FaceForensics++](https://github.com/ondyari/FaceForensics), [DFDC](https://ai.meta.com/datasets/dfdc/)).

- **Dataset bias.** The two auto-download sources (randomuser.me vs thispersondoesnotexist.com) also differ in JPEG compression level, colour temperature, and background complexity. The model can exploit these distribution differences instead of genuine facial manipulation signals. Use the Grad-CAM heatmaps to diagnose this — if highlighted regions fall on backgrounds or image borders rather than faces, the model is learning the wrong signal.

- **CPU speed.** ViT-Small/16 is slower than CNNs on CPU due to the attention mechanism's quadratic complexity over patches.

- **Binary classification only.** The system detects Real vs Fake but does not identify the manipulation type (face swap, expression synthesis, full synthesis, etc.).
