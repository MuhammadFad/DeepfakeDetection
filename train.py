"""
Fine-tuning pipeline — two-phase training for all three deepfake detection models.

Phase 1 (warmup): Freeze backbone, train only the classification head.
Phase 2 (fine-tune): Unfreeze everything, use differential learning rates.

Usage:
    python train.py                    # train all three models
    python train.py --model xception   # train one model
    python train.py --epochs 15        # override epoch count
"""

import argparse
import gc
import json
import os
import sys

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    DEVICE, NUM_CLASSES,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET, MODEL_DISPLAY_NAMES,
    CHECKPOINT_DIR, OUTPUT_DIR, TRAINING_HISTORY_PATH,
    BATCH_SIZE_TRAIN, WARMUP_EPOCHS, TOTAL_EPOCHS,
    LR_HEAD, LR_BACKBONE, LR_HEAD_FT, WEIGHT_DECAY, LABEL_SMOOTHING,
)
from dataset import get_dataloader, DeepfakeDataset
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def split_params(model) -> tuple[list, list]:
    """Return (backbone_params, head_params) for differential LR."""
    try:
        head = model.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        head_ps  = [p for p in model.parameters() if id(p) in head_ids]
        return backbone, head_ps
    except Exception:
        return [], list(model.parameters())


def freeze_backbone(model):
    backbone, _ = split_params(model)
    for p in backbone:
        p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc='    batch', leave=False, ncols=80)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += bs
        pbar.set_postfix(loss=f'{loss.item():.3f}', acc=f'{100.*correct/total:.1f}%')

    return total_loss / total, 100. * correct / total


def eval_epoch(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            total_loss += criterion(out, labels).item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100. * correct / total


# ---------------------------------------------------------------------------
# Per-model training
# ---------------------------------------------------------------------------

def train_model(model_name: str, warmup_epochs: int, total_epochs: int) -> dict:
    display = MODEL_DISPLAY_NAMES[model_name]
    fine_tune_epochs = total_epochs - warmup_epochs

    print(f'\n{"=" * 56}')
    print(f'  Training: {display}')
    print(f'{"=" * 56}')

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader = get_dataloader('train', model_name, batch_size=BATCH_SIZE_TRAIN)
    val_loader   = get_dataloader('val',   model_name, batch_size=BATCH_SIZE_TRAIN)
    has_val      = len(val_loader.dataset) > 0

    print(f'  Train: {len(train_loader.dataset)} images  |  '
          f'Val: {len(val_loader.dataset) if has_val else 0} images')

    if len(train_loader.dataset) == 0:
        print('  No training data found. Run:  python download_data.py --count 50 --split')
        return {}

    # ── Model ────────────────────────────────────────────────────────────────
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')

    # ── Phase 1: Warmup (head only) ──────────────────────────────────────────
    print(f'\n  [Phase 1] Warmup — head only ({warmup_epochs} epochs)')
    freeze_backbone(model)
    _, head_params = split_params(model)
    trainable = head_params if head_params else list(model.parameters())
    optimizer = optim.AdamW(trainable, lr=LR_HEAD)

    for ep in range(warmup_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion) if has_val else (0., tr_acc)
        history['train_loss'].append(round(tr_loss, 4))
        history['train_acc'].append(round(tr_acc, 2))
        history['val_loss'].append(round(vl_loss, 4))
        history['val_acc'].append(round(vl_acc, 2))
        tag = ''
        if has_val and vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': vl_acc, 'epoch': ep + 1}, ckpt_path)
            tag = '  [saved]'
        print(f'  Ep {ep+1:2d}/{total_epochs} | '
              f'train {tr_loss:.4f}/{tr_acc:5.1f}% | '
              f'val {vl_acc:5.1f}%{tag}')

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────────
    print(f'\n  [Phase 2] Fine-tune — full network ({fine_tune_epochs} epochs)')
    unfreeze_all(model)
    backbone_params, head_params = split_params(model)
    param_groups = [{'params': backbone_params, 'lr': LR_BACKBONE},
                    {'params': head_params,     'lr': LR_HEAD_FT}]
    optimizer  = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fine_tune_epochs, eta_min=1e-7)

    for ep in range(fine_tune_epochs):
        global_ep = warmup_epochs + ep + 1
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion) if has_val else (0., tr_acc)
        history['train_loss'].append(round(tr_loss, 4))
        history['train_acc'].append(round(tr_acc, 2))
        history['val_loss'].append(round(vl_loss, 4))
        history['val_acc'].append(round(vl_acc, 2))
        scheduler.step()

        tag = ''
        if vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': vl_acc, 'epoch': global_ep}, ckpt_path)
            tag = '  [saved]'
        print(f'  Ep {global_ep:2d}/{total_epochs} | '
              f'train {tr_loss:.4f}/{tr_acc:5.1f}% | '
              f'val {vl_acc:5.1f}%{tag}')

    print(f'\n  Best val accuracy: {best_val_acc:.1f}%  ->  {ckpt_path}')
    del model, optimizer
    gc.collect()
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fine-tune deepfake detection models')
    parser.add_argument('--model', choices=[MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET],
                        default=None, help='Train a single model (default: all three)')
    parser.add_argument('--epochs', type=int, default=TOTAL_EPOCHS,
                        help=f'Total epochs (default: {TOTAL_EPOCHS})')
    parser.add_argument('--warmup', type=int, default=WARMUP_EPOCHS,
                        help=f'Warmup epochs (default: {WARMUP_EPOCHS})')
    args = parser.parse_args()

    models_to_train = ([args.model] if args.model
                       else [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET])

    print(f'\nDevice: {DEVICE}')
    print(f'Epochs: {args.epochs}  (warmup={args.warmup}, fine-tune={args.epochs - args.warmup})')

    all_history = {}
    for mn in models_to_train:
        hist = train_model(mn, warmup_epochs=args.warmup, total_epochs=args.epochs)
        if hist:
            all_history[mn] = hist
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined history for report.py
    if all_history:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Merge with any existing history so partial runs accumulate
        existing = {}
        if os.path.exists(TRAINING_HISTORY_PATH):
            try:
                with open(TRAINING_HISTORY_PATH, encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update(all_history)
        with open(TRAINING_HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)
        print(f'\nTraining history saved to: {TRAINING_HISTORY_PATH}')

    print('\nDone! Run  python main.py  to evaluate with trained weights.\n')


if __name__ == '__main__':
    main()
