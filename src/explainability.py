"""
Grad-CAM heatmap generation for model explainability.

For CNN models (XceptionNet, EfficientNet): hooks into the last Conv2d layer.
For ViT: hooks into the last transformer block's norm with patch reshaping.
"""

import base64
import io
import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from .config import IMAGE_SIZE, VIT_IMAGE_SIZE


def available() -> bool:
    return _AVAILABLE


def _target_layer(model, model_name: str):
    """Return CAM target layer: last transformer norm for ViT, last Conv2d for CNNs."""
    if 'vit' in model_name:
        return model.blocks[-1].norm1
    last_conv = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


def _vit_reshape(tensor, height=14, width=14):
    """Reshape ViT patch tokens (B, N+1, D) → (B, D, H, W) for GradCAM."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def encode_image(img_path, size: int = 224) -> str | None:
    """Return base64 JPEG of the image resized to size×size, or None on failure."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (size, size))
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        return None


def generate_heatmap(model, model_name: str, tensor: torch.Tensor, img_path) -> str | None:
    """
    Run Grad-CAM on the model's predicted class and return a base64 JPEG
    of the heatmap overlaid on the original image. Returns None on failure.

    Red/warm regions = high influence on the prediction.
    Blue/cool regions = low influence.
    """
    if not _AVAILABLE:
        return None
    try:
        layer = _target_layer(model, model_name)
        if layer is None:
            logger.warning("GradCAM: no target layer found for %s", model_name)
            return None

        # Ensure tensor is on the same device as the model
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        reshape = _vit_reshape if 'vit' in model_name else None

        with torch.no_grad():
            pred_class = model(tensor).argmax(1).item()

        cam = GradCAM(model=model, target_layers=[layer], reshape_transform=reshape)
        grayscale = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0]

        size = VIT_IMAGE_SIZE if 'vit' in model_name else IMAGE_SIZE
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (size, size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        vis = show_cam_on_image(img_rgb, grayscale, use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(vis).save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as exc:
        logger.warning("GradCAM failed for %s: %s", model_name, exc)
        return None
