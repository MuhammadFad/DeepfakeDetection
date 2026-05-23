import logging
import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

_transform_cache: dict = {}


def _get_transform(target_size: int) -> transforms.Compose:
    if target_size not in _transform_cache:
        _transform_cache[target_size] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return _transform_cache[target_size]


def load_and_preprocess(image_path, target_size: int = None) -> torch.Tensor | None:
    """Load an image from disk and return a (1, 3, H, W) tensor ready for inference."""
    if target_size is None:
        target_size = IMAGE_SIZE

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Could not read image (skipping): %s", image_path)
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        transform = _get_transform(target_size)
        tensor = transform(img)          # (3, H, W)
        tensor = tensor.unsqueeze(0)     # (1, 3, H, W)
        return tensor

    except Exception as exc:
        logger.warning("Preprocessing failed for %s: %s", image_path, exc)
        return None
