"""
Pass-split parallel training pipeline.

Pass 1 — XceptionNet alone (needs headroom for activation spikes).
Pass 2 — ViT + EfficientNet together (interleaved epoch loop, shared cache).

One cache fill. All models share the same 299x299 GPUCachedDataset.
ViT resizes 299->224 on-GPU inside run_one_epoch(). EfficientNet uses 299 natively.

Fallback: scripts/train.py for sequential single-model training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import math
import os
import platform
import time
from datetime import datetime

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import src.logger as logger
from src.config import (
    DEVICE, NUM_CLASSES, CPU_THREADS,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET, MODEL_DISPLAY_NAMES,
    CHECKPOINT_DIR, OUTPUT_DIR, TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS, TOTAL_EPOCHS, EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING,
)
from src.dataset import GPUCachedDataset, get_dataloader

if CPU_THREADS > 0:
    torch.set_num_threads(CPU_THREADS)

torch.backends.cudnn.benchmark = True

PROFILE_LOG = os.path.join(OUTPUT_DIR, 'profile_log.txt')


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def _is_valid_checkpoint(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        torch.load(path, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def load_and_prepare_model(model_name: str) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=True)
    model = torch.compile(model, mode='reduce-overhead')
    return model


def build_optimizer(model: nn.Module, phase: int) -> optim.Optimizer:
    fused = torch.cuda.is_available()
    if phase == 1:
        for p in model.parameters():
            p.requires_grad = False
        try:
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True
        except Exception:
            for p in model.parameters():
                p.requires_grad = True
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            fused=fused,
        )
    else:
        for p in model.parameters():
            p.requires_grad = True
        backbone_params = [p for n, p in model.named_parameters()
                           if 'head' not in n and 'classifier' not in n]
        head_params     = [p for n, p in model.named_parameters()
                           if 'head' in n or 'classifier' in n]
        if not head_params:
            head_params, backbone_params = backbone_params, []
        return optim.AdamW(
            [{'params': backbone_params, 'lr': 1e-5},
             {'params': head_params,     'lr': 1e-4}],
            fused=fused,
        )


def make_state(model: nn.Module, optimizer: optim.Optimizer, scaler) -> dict:
    return {
        'model':          model,
        'optimizer':      optimizer,
        'scaler':         scaler,
        'best_val_loss':  float('inf'),
        'best_val_acc':   0.0,
        'patience_count': 0,
        'best_epoch':     0,
        'history':        [],
    }


def save_checkpoint(state: dict, model_name: str, epoch: int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        'model_state_dict': state['model'].state_dict(),
        'epoch':            epoch,
        'val_acc':          state['best_val_acc'],
        'val_loss':         state['best_val_loss'],
        'history':          state['history'],
    }
    tmp  = Path(CHECKPOINT_DIR) / f'{model_name}_best.tmp'
    dest = Path(CHECKPOINT_DIR) / f'{model_name}_best.pth'
    torch.save(ckpt, tmp)
    tmp.rename(dest)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def run_one_epoch(model_name: str, state: dict,
                  train_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model     = state['model']
    optimizer = state['optimizer']
    scaler    = state['scaler']
    model.train()
    total_loss = total_correct = total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        # ViT requires 224x224 — cache is 299x299, resize on GPU
        if model_name == MODEL_VIT:
            inputs = F.interpolate(
                inputs.float(), size=(224, 224),
                mode='bilinear', align_corners=False,
            ).half()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss    += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples * 100


def run_val_epoch(model_name: str, state: dict,
                  val_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model = state['model']
    model.eval()
    total_loss = total_correct = total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            total_loss    += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += inputs.size(0)

    model.train()
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples * 100


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

def run_profiled_steps(model_name: str, state: dict,
                       train_loader: DataLoader, criterion: nn.Module, batch_size: int):
    model     = state['model']
    optimizer = state['optimizer']
    scaler    = state['scaler']
    step_times: list[float] = []
    active_start = 5  # wait=2 + warmup=3

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=15),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        model.train()
        data_iter = iter(train_loader)
        for step_idx in range(20):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, labels = next(data_iter)

            in_active = step_idx >= active_start
            if in_active:
                t0 = time.perf_counter()

            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if model_name == MODEL_VIT:
                inputs = F.interpolate(
                    inputs.float(), size=(224, 224),
                    mode='bilinear', align_corners=False,
                ).half()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if in_active:
                torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1000)

            prof.step()

    dataset_size = len(train_loader.dataset)
    avg = sum(step_times) / len(step_times) if step_times else 0.0
    mn  = min(step_times) if step_times else 0.0
    mx  = max(step_times) if step_times else 0.0
    estimated_epoch = math.ceil(dataset_size / batch_size) * avg / 1000

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROFILE_LOG, 'a', encoding='utf-8') as f:
        f.write(f'\n{"=" * 60}\n')
        f.write(f'Profile — {model_name} — {now}\n')
        f.write(f'{"=" * 60}\n\n')
        f.write('--- TOP 20 OPS BY CUDA TIME ---\n')
        f.write(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
        f.write('\n\n--- TOP 20 OPS BY CPU TIME ---\n')
        f.write(prof.key_averages().table(sort_by='cpu_time_total', row_limit=20))
        f.write('\n\n--- MEMORY SUMMARY ---\n')
        f.write(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=10))
        f.write('\n\n--- STEP TIMING ---\n')
        f.write(f'Average step time : {avg:.1f} ms\n')
        f.write(f'Fastest step      : {mn:.1f} ms\n')
        f.write(f'Slowest step      : {mx:.1f} ms\n')
        f.write(f'Estimated epoch   : {estimated_epoch:.1f} sec\n')
        f.write('\n--- VRAM ---\n')
        f.write(f'Peak allocated    : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n')
        f.write(f'Current allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB\n')
        f.write(f'{"=" * 60}\n')
    print(f'  Profile written to {PROFILE_LOG}')


# ---------------------------------------------------------------------------
# Core pass runner
# ---------------------------------------------------------------------------

def run_pass(model_names: list[str], train_dataset: GPUCachedDataset,
             val_loaders: dict, criterion: nn.Module,
             batch_size: int, warmup: int, total_epochs: int,
             use_profile: bool = False):
    fine_tune_epochs = total_epochs - warmup

    # Load all models for this pass
    active: dict[str, dict] = {}
    for name in model_names:
        model     = load_and_prepare_model(name)
        optimizer = build_optimizer(model, phase=1)
        scaler    = torch.amp.GradScaler('cuda')
        active[name] = make_state(model, optimizer, scaler)
        print(f'  Loaded {MODEL_DISPLAY_NAMES[name]}')

    print(f'  VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    # Profile — XceptionNet only, before epoch loop
    if use_profile and MODEL_XCEPTION in active:
        print(f'  Profiling {MODEL_DISPLAY_NAMES[MODEL_XCEPTION]}...')
        run_profiled_steps(MODEL_XCEPTION, active[MODEL_XCEPTION],
                           train_loader, criterion, batch_size)

    # ===== Phase 1: Warmup — head only =====
    print(f'\n  [Phase 1] Warmup — head only ({warmup} epochs)')
    logger.log('[Phase 1] warmup — head only')

    for epoch in range(1, warmup + 1):
        for name, state in list(active.items()):
            tr_loss, tr_acc = run_one_epoch(name, state, train_loader, criterion)
            vl_loss, vl_acc = run_val_epoch(name, state, val_loaders[name], criterion)
            state['history'].append({
                'phase': 1, 'epoch': epoch,
                'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 2),
                'val_loss':   round(vl_loss, 4), 'val_acc':   round(vl_acc, 2),
            })
            tag = ''
            if vl_loss < state['best_val_loss']:
                state['best_val_loss'] = vl_loss
                state['best_val_acc']  = vl_acc
                state['best_epoch']    = epoch
                save_checkpoint(state, name, epoch)
                tag = '  [saved]'
            disp = MODEL_DISPLAY_NAMES[name][:14]
            line = (f'[{disp:14s}] ep{epoch:2d}/{total_epochs} '
                    f'train {tr_acc:.1f}% loss {tr_loss:.4f} | '
                    f'val {vl_acc:.1f}% loss {vl_loss:.4f}{tag}')
            print(f'  {line}')
            logger.log(line)

    # ===== Phase 2: Full fine-tune =====
    torch.cuda.empty_cache()
    p2_batch = max(8, batch_size // 2)
    train_loader_p2 = DataLoader(
        train_dataset, batch_size=p2_batch, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    print(f'\n  [Phase 2] Fine-tune — full network '
          f'({fine_tune_epochs} epochs, patience={EARLY_STOP_PATIENCE}, batch={p2_batch})')
    logger.log(f'[Phase 2] full fine-tune  patience={EARLY_STOP_PATIENCE}  batch={p2_batch}')

    for name, state in active.items():
        state['optimizer'] = build_optimizer(state['model'], phase=2)
        state['scheduler'] = optim.lr_scheduler.CosineAnnealingLR(
            state['optimizer'], T_max=fine_tune_epochs, eta_min=1e-7)
        state['patience_count'] = 0

    for epoch in range(1, fine_tune_epochs + 1):
        global_ep = warmup + epoch
        to_remove: list[str] = []

        for name, state in list(active.items()):
            tr_loss, tr_acc = run_one_epoch(name, state, train_loader_p2, criterion)
            vl_loss, vl_acc = run_val_epoch(name, state, val_loaders[name], criterion)
            state['scheduler'].step()
            state['history'].append({
                'phase': 2, 'epoch': epoch,
                'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 2),
                'val_loss':   round(vl_loss, 4), 'val_acc':   round(vl_acc, 2),
            })
            tag = ''
            if vl_loss < state['best_val_loss']:
                state['best_val_loss']  = vl_loss
                state['best_val_acc']   = vl_acc
                state['best_epoch']     = global_ep
                state['patience_count'] = 0
                save_checkpoint(state, name, global_ep)
                tag = '  [saved]'
            else:
                state['patience_count'] += 1

            disp = MODEL_DISPLAY_NAMES[name][:14]
            line = (f'[{disp:14s}] ep{global_ep:2d}/{total_epochs} '
                    f'train {tr_acc:.1f}% loss {tr_loss:.4f} | '
                    f'val {vl_acc:.1f}% loss {vl_loss:.4f}{tag}')
            print(f'  {line}')
            logger.log(line)

            if state['patience_count'] >= EARLY_STOP_PATIENCE:
                es_line = f'[{disp:14s}] early stop at epoch {global_ep}'
                print(f'  {es_line}')
                logger.log(es_line)
                to_remove.append(name)

        for name in to_remove:
            if name in active:
                st = active.pop(name)
                del val_loaders[name]
                model_ref = st.pop('model')
                del model_ref
                del st
                gc.collect()
                torch.cuda.empty_cache()
                print(f'  VRAM after unloading {name}: '
                      f'{torch.cuda.memory_allocated()/1e9:.2f} GB')

        if not active:
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Pass-split parallel deepfake detection training')
    p.add_argument('--epochs',        type=int, default=TOTAL_EPOCHS)
    p.add_argument('--warmup',        type=int, default=WARMUP_EPOCHS)
    p.add_argument('--auto-scale',    action='store_true')
    p.add_argument('--cache',         action='store_true',
                   help='(accepted for compatibility — GPU cache is always used)')
    p.add_argument('--profile',       action='store_true')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--force',         action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    skip = args.skip_existing and not args.force

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.init(os.path.join(OUTPUT_DIR, 'training_log.txt'))

    print('Pass-split parallel training')
    print(f'Device: {DEVICE}')
    print(f'Epochs: {args.epochs}  (warmup={args.warmup}, '
          f'fine-tune={args.epochs - args.warmup})')

    # VRAM check
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_gb  = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()) / 1e9
        print(f'VRAM: {free_gb:.1f} GB free of {total_gb:.1f} GB total')
        if total_gb < 14.0:
            print('WARNING: less than 14 GB VRAM detected')
            print('This script is tuned for a T4 (15 GB) — consider train.py instead')

    # Batch size via auto-scale (snap to nearest multiple of 32)
    if args.auto_scale and torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        raw_batch = int(vram_mb * 0.90 // 75)
        batch_size = max(32, (raw_batch // 32) * 32)
        print(f'[Auto-Scale] GPU VRAM: {vram_mb:.0f} MB -> batch={batch_size}')
    else:
        batch_size = 96

    val_workers = 0 if platform.system() == 'Windows' else 2

    # Determine which models to train
    def should_train(mn: str) -> bool:
        if not skip:
            return True
        return not _is_valid_checkpoint(os.path.join(CHECKPOINT_DIR, f'{mn}_best.pth'))

    pass1_models = [MODEL_XCEPTION] if should_train(MODEL_XCEPTION) else []
    pass2_models = [mn for mn in [MODEL_VIT, MODEL_EFFICIENTNET] if should_train(mn)]

    if not pass1_models and not pass2_models:
        print('All models have valid checkpoints. Use --force to retrain.')
        logger.close()
        return

    logger.log(f'parallel train  pass1={pass1_models}  pass2={pass2_models}  '
               f'batch={batch_size}  epochs={args.epochs}')

    # Fill cache ONCE at 299x299 — shared by all three models
    print('\nFilling GPU cache...')
    train_dataset = GPUCachedDataset(
        split='train',
        model_name=MODEL_XCEPTION,  # 299x299; ViT resizes on-GPU in run_one_epoch
        device=DEVICE,
    )
    print(f'Cache loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM used')

    # Val loaders — CPU path, one per model (each uses its own target size)
    all_needed = pass1_models + pass2_models
    val_loaders_all = {
        mn: get_dataloader('val', mn, batch_size=batch_size, num_workers=val_workers)
        for mn in all_needed
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Pass 1 — XceptionNet alone
    if pass1_models:
        print(f'\n{"=" * 56}')
        print(f'  PASS 1 — {MODEL_DISPLAY_NAMES[MODEL_XCEPTION]}')
        print(f'{"=" * 56}')
        val_loaders_p1 = {mn: val_loaders_all[mn] for mn in pass1_models}
        run_pass(pass1_models, train_dataset, val_loaders_p1, criterion,
                 batch_size=batch_size, warmup=args.warmup, total_epochs=args.epochs,
                 use_profile=args.profile)
        torch.cuda.empty_cache()
        print(f'\nVRAM after pass 1: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    # Pass 2 — ViT + EfficientNet together
    if pass2_models:
        display = ' + '.join(MODEL_DISPLAY_NAMES[m] for m in pass2_models)
        print(f'\n{"=" * 56}')
        print(f'  PASS 2 — {display}')
        print(f'{"=" * 56}')
        val_loaders_p2 = {mn: val_loaders_all[mn] for mn in pass2_models}
        run_pass(pass2_models, train_dataset, val_loaders_p2, criterion,
                 batch_size=batch_size, warmup=args.warmup, total_epochs=args.epochs,
                 use_profile=False)

    # Save training history to shared JSON (compatible with train.py output)
    all_history: dict = {}
    for mn in all_needed:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'{mn}_best.pth')
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if 'history' in ckpt:
                    all_history[mn] = ckpt['history']
            except Exception:
                pass

    if all_history:
        existing: dict = {}
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

    print('\nAll models trained.')
    logger.log('parallel train complete')
    logger.close()


if __name__ == '__main__':
    main()
