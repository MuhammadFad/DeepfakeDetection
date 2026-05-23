import os
import sys
import logging

import torch
import timm

from .config import (
    DEVICE, NUM_CLASSES,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET,
    MODEL_DISPLAY_NAMES, CHECKPOINT_DIR,
)

logger = logging.getLogger(__name__)


def load_model(model_name: str):
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
        model = model.to(DEVICE)
        model.eval()
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        logger.info("Loaded: %s", display)
        return model
    except Exception as exc:
        logger.error("Failed to load %s: %s", model_name, exc)
        return None


def load_checkpoint(model_name: str):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state['model_state_dict'])
            val_acc = state.get('val_acc', '?')
            epoch   = state.get('epoch', '?')
            print(f'checkpoint (ep={epoch}, val={val_acc:.1f}%)', end=' ')
        except Exception as exc:
            logger.warning("Could not load checkpoint %s: %s", ckpt_path, exc)
            print('checkpoint FAILED, using ImageNet weights', end=' ')
    else:
        print('ImageNet weights (no checkpoint)', end=' ')

    model.eval()
    return model


def load_all_models(use_checkpoints: bool = True) -> dict:
    models = {}
    for name in [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET]:
        display = MODEL_DISPLAY_NAMES.get(name, name)
        print(f"  {display}... ", end='', flush=True)
        try:
            model = load_checkpoint(name) if use_checkpoints else load_model(name)
        except Exception as exc:
            print("FAILED")
            print(f"\nCould not load '{name}': {exc}")
            print("Make sure timm is installed:  pip install timm")
            sys.exit(1)
        print("OK")
        models[name] = model
    return models


def run_inference(model, tensor: torch.Tensor) -> tuple[str | None, float | None]:
    try:
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        label = 'Real' if predicted.item() == 0 else 'Fake'
        conf_pct = confidence.item() * 100.0
        return label, conf_pct

    except Exception as exc:
        logger.warning("Inference error: %s", exc)
        return None, None
