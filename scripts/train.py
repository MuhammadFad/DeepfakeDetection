"""
Fine-tuning pipeline — two-phase training for all three deepfake detection models.

Phase 1 (warmup): Freeze backbone, train only the classification head.
Phase 2 (fine-tune): Unfreeze everything, use differential learning rates.

Recommended: run via train_colab.ipynb on a Colab T4 GPU.
For local runs with a GPU, call from the project root:
    python scripts/train.py --model xception --auto-scale
    python scripts/train.py --model vit_small_patch16_224 --auto-scale
    python scripts/train.py --model efficientnet_b4 --auto-scale
    python scripts/train.py --skip-existing
    python scripts/train.py --epochs 15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import os
import platform

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import src.logger as logger
from src.config import (
    DEVICE, NUM_CLASSES, CPU_THREADS,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET, MODEL_DISPLAY_NAMES,
    CHECKPOINT_DIR, OUTPUT_DIR, TRAINING_HISTORY_PATH,
    BATCH_SIZE_TRAIN, WARMUP_EPOCHS, TOTAL_EPOCHS, EARLY_STOP_PATIENCE,
    LR_HEAD, LR_BACKBONE, LR_HEAD_FT, WEIGHT_DECAY, LABEL_SMOOTHING,
)

if CPU_THREADS > 0:
    torch.set_num_threads(CPU_THREADS)

torch.backends.cudnn.benchmark = True

from src.dataset import get_dataloader

# ---------------------------------------------------------------------------
# Dynamic Resource Allocator
# ---------------------------------------------------------------------------

def get_dynamic_resources(auto_scale=False, default_batch=16, default_workers=0):
    """
    Two-way resource allocator.
    If False -> strict adherence to stingy defaults.
    If True  -> detects hardware and consumes ~90% of available capacity.
    """
    if not auto_scale:
        return default_batch, default_workers

    print("\n[Auto-Scale] 🚀 Hardware detected. Calculating optimal load...")

    # 1. CPU Workers
    cores = os.cpu_count() or 2
    if platform.system() == 'Windows':
        opt_workers = 0 
        print(f"[Auto-Scale] Windows OS detected. Forcing workers to 0 to prevent crashes.")
    else:
        opt_workers = max(1, cores - 1) if cores > 2 else cores
        print(f"[Auto-Scale] CPU Cores: {cores} -> Allocating {opt_workers} workers.")

    # 2. GPU Batch Size
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_mb = vram_bytes / (1024 ** 2)
        target_vram_mb = vram_mb * 0.90
        
        mb_per_image = 75 
        raw_batch_size = int(target_vram_mb // mb_per_image)

        safe_multiples = [128, 96, 64, 32, 16, 8]
        opt_batch = 16
        for p in safe_multiples:
            if raw_batch_size >= p:
                opt_batch = p
                break
                
        print(f"[Auto-Scale] GPU VRAM: {vram_mb:.0f} MB -> Targeting {target_vram_mb:.0f} MB")
        print(f"[Auto-Scale] Calculated Max Batch: {raw_batch_size} -> Snapping to {opt_batch}")
    else:
        opt_batch = default_batch
        print("[Auto-Scale] No GPU detected. Falling back to default batch size.")

    return opt_batch, opt_workers


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def split_params(model) -> tuple[list, list]:
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

def train_epoch(model, loader, optimizer, criterion, scaler) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc='    batch', leave=False, ncols=80)
    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
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

def _is_valid_checkpoint(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        torch.load(path, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False

def _save_checkpoint(state: dict, path: str):
    import stat
    tmp = path + '.tmp'
    torch.save(state, tmp)
    if os.path.exists(path):
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)
    os.rename(tmp, path)

def train_model(model_name: str, warmup_epochs: int, total_epochs: int,
                skip_existing: bool, batch_size: int, num_workers: int,
                use_cache: bool = False) -> dict:
    display = MODEL_DISPLAY_NAMES[model_name]
    fine_tune_epochs = total_epochs - warmup_epochs
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pth')

    tmp_path = ckpt_path + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f'\n{"=" * 56}')
    print(f'  Training: {display}')
    print(f'{"=" * 56}')

    if skip_existing and _is_valid_checkpoint(ckpt_path):
        msg = f'{display}: valid checkpoint exists — skipping'
        print(f'  Valid checkpoint already exists — skipping.')
        print(f'  (Use --force to retrain anyway)')
        logger.log(msg)
        return {}

    logger.log(f'--- {display} ---')
    logger.log(f'warmup={warmup_epochs}  total={total_epochs}  patience={EARLY_STOP_PATIENCE}  batch={batch_size}')

    eff_workers  = 0 if use_cache else num_workers
    train_loader = get_dataloader('train', model_name, batch_size=batch_size, num_workers=eff_workers, use_cache=use_cache, device=DEVICE)
    val_loader   = get_dataloader('val',   model_name, batch_size=batch_size, num_workers=num_workers)
    has_val      = len(val_loader.dataset) > 0

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset) if has_val else 0
    print(f'  Train: {n_train} images  |  Val: {n_val} images')
    logger.log(f'data  train={n_train}  val={n_val}')

    if n_train == 0:
        print('  No training data found — populate images/train/ first.')
        logger.log('ABORT: no training data found')
        return {}

    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device='cuda', enabled=torch.cuda.is_available())
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    def _record(tr_loss, tr_acc, vl_loss, vl_acc):
        history['train_loss'].append(round(tr_loss, 4))
        history['train_acc'].append(round(tr_acc, 2))
        history['val_loss'].append(round(vl_loss, 4))
        history['val_acc'].append(round(vl_acc, 2))

    # Phase 1: Warmup
    print(f'\n  [Phase 1] Warmup — head only ({warmup_epochs} epochs)')
    logger.log(f'[Phase 1] warmup — head only')
    freeze_backbone(model)
    _, head_params = split_params(model)
    trainable = head_params if head_params else list(model.parameters())
    optimizer = optim.AdamW(trainable, lr=LR_HEAD)

    for ep in range(warmup_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion) if has_val else (0., tr_acc)
        _record(tr_loss, tr_acc, vl_loss, vl_acc)
        tag = ''
        if has_val and vl_loss < best_val_loss:
            best_val_loss = vl_loss
            _save_checkpoint({'model_state_dict': model.state_dict(),
                               'val_loss': vl_loss, 'val_acc': vl_acc, 'epoch': ep + 1}, ckpt_path)
            tag = '  [saved]'
        line = (f'Ep {ep+1:2d}/{total_epochs} | train {tr_loss:.4f}/{tr_acc:5.1f}% | val {vl_acc:5.1f}% (loss {vl_loss:.4f}){tag}')
        print(f'  {line}')
        logger.log(line)

    # Phase 2: Full fine-tune — halve batch size; all params have gradients now
    p2_batch = max(8, batch_size // 2)
    if use_cache:
        # Reuse the cached GPU dataset — no second cache fill, no extra VRAM consumed
        from torch.utils.data import DataLoader as _DL
        train_loader = _DL(train_loader.dataset, batch_size=p2_batch, shuffle=True,
                           num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        train_loader = get_dataloader('train', model_name, batch_size=p2_batch,
                                      num_workers=eff_workers)
    val_loader = get_dataloader('val', model_name, batch_size=p2_batch, num_workers=num_workers)
    print(f'\n  [Phase 2] Fine-tune — full network ({fine_tune_epochs} epochs, patience={EARLY_STOP_PATIENCE}, batch={p2_batch})')
    logger.log(f'[Phase 2] full fine-tune  patience={EARLY_STOP_PATIENCE}  batch={p2_batch}')
    no_improve = 0
    unfreeze_all(model)
    backbone_params, head_params = split_params(model)
    param_groups = [{'params': backbone_params, 'lr': LR_BACKBONE},
                    {'params': head_params,     'lr': LR_HEAD_FT}]
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)

    for ep in range(fine_tune_epochs):
        global_ep = warmup_epochs + ep + 1
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion) if has_val else (0., tr_acc)
        _record(tr_loss, tr_acc, vl_loss, vl_acc)
        scheduler.step()

        tag = ''
        if has_val and vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve = 0
            _save_checkpoint({'model_state_dict': model.state_dict(),
                               'val_loss': vl_loss, 'val_acc': vl_acc, 'epoch': global_ep}, ckpt_path)
            tag = '  [saved]'
        elif has_val:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                line = (f'Ep {global_ep:2d}/{total_epochs} | train {tr_loss:.4f}/{tr_acc:5.1f}% | val {vl_acc:5.1f}% (loss {vl_loss:.4f})  [early stop]')
                print(f'  {line}')
                logger.log(line)
                break

        line = (f'Ep {global_ep:2d}/{total_epochs} | train {tr_loss:.4f}/{tr_acc:5.1f}% | val {vl_acc:5.1f}% (loss {vl_loss:.4f}){tag}')
        print(f'  {line}')
        logger.log(line)

    best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
    print(f'\n  Best val accuracy: {best_acc:.1f}%  ->  {ckpt_path}')
    logger.log(f'best val acc: {best_acc:.1f}%  ->  {ckpt_path}')
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
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip models that already have a valid checkpoint')
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if a valid checkpoint exists')
    parser.add_argument('--auto-scale', action='store_true',
                        help='Automatically maximize batch size and CPU workers for the current hardware')
    parser.add_argument('--cache', action='store_true',
                        help='Cache entire training set on GPU VRAM as FP16 (requires CUDA)')
    args = parser.parse_args()

    skip = args.skip_existing and not args.force
    models_to_train = ([args.model] if args.model else [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.init(os.path.join(OUTPUT_DIR, 'training_log.txt'))

    print(f'\nDevice: {DEVICE}')
    print(f'Epochs: {args.epochs}  (warmup={args.warmup}, fine-tune={args.epochs - args.warmup})')

    # Calculate resources!
    dynamic_batch, dynamic_workers = get_dynamic_resources(
        auto_scale=args.auto_scale, 
        default_batch=BATCH_SIZE_TRAIN,  
        default_workers=0                
    )

    logger.log(f'device={DEVICE} epochs={args.epochs} warmup={args.warmup} batch={dynamic_batch} workers={dynamic_workers}')
    logger.log(f'models: {", ".join(models_to_train)}')

    all_history = {}
    for mn in models_to_train:
        hist = train_model(mn, warmup_epochs=args.warmup, total_epochs=args.epochs,
                           skip_existing=skip, batch_size=dynamic_batch, num_workers=dynamic_workers,
                           use_cache=args.cache)
        if hist:
            all_history[mn] = hist
        gc.collect()
        torch.cuda.empty_cache()

    if all_history:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    logger.log('all models done')
    logger.close()
    print('\nDone! Run  python main.py  to evaluate with trained weights.\n')


if __name__ == '__main__':
    main()