import os
import torch

# CPU thread limit — prevents PyTorch from saturating all cores during training,
# which causes thermal throttling and crashes on laptops.
# Set to 0 to let PyTorch decide (uses all cores).
CPU_THREADS = max(1, (os.cpu_count() or 4) // 2)

# Image settings
IMAGE_SIZE = 299          # XceptionNet + EfficientNet input size
VIT_IMAGE_SIZE = 224      # ViT requires 224x224
BATCH_SIZE = 1
CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES = 2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths — BASE_DIR is the project root (one level above this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_HTML = os.path.join(OUTPUT_DIR, 'results.html')

# Split-based training directories
TRAIN_DIR = os.path.join(BASE_DIR, 'images', 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'images', 'val')
TEST_DIR  = os.path.join(BASE_DIR, 'images', 'test')

# Checkpoints + training history
CHECKPOINT_DIR        = os.path.join(BASE_DIR, 'checkpoints')
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, 'training_history.json')

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Training hyper-parameters
BATCH_SIZE_TRAIN    = 64
WARMUP_EPOCHS       = 4      # Phase 1: train head only
TOTAL_EPOCHS        = 20     # Phase 1 + Phase 2 combined (early stopping ends it sooner)
EARLY_STOP_PATIENCE = 5      # Stop if val loss doesn't improve for this many epochs
LR_HEAD             = 1e-3   # Phase 1 head LR
LR_BACKBONE         = 1e-5   # Phase 2 backbone LR (differential)
LR_HEAD_FT          = 1e-4   # Phase 2 head LR (differential)
WEIGHT_DECAY        = 1e-4
LABEL_SMOOTHING     = 0.1

# Model identifiers (timm names)
MODEL_XCEPTION     = 'xception'
MODEL_VIT          = 'vit_small_patch16_224'   # Small (22M) beats Base (86M) on small datasets
MODEL_EFFICIENTNET = 'efficientnet_b4'

MODEL_DISPLAY_NAMES = {
    MODEL_XCEPTION:     'XceptionNet (CNN)',
    MODEL_VIT:          'ViT-Small/16 (Transformer)',
    MODEL_EFFICIENTNET: 'EfficientNet-B4 (Hybrid)',
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Labels — index 0 = Real, index 1 = Fake
LABEL_REAL = 'Real'
LABEL_FAKE = 'Fake'
LABELS = [LABEL_REAL, LABEL_FAKE]
