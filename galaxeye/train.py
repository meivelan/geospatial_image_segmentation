"""
train.py
--------
Training loop, early stopping, and checkpoint management.

Usage
-----
    from galaxeye.train import train
    best_f1 = train(train_loader, val_loader)

    # Or resume from a previous run:
    best_f1 = train(train_loader, val_loader, resume=True)
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CONFIG
from .model  import get_model, compute_metrics, compute_confusion_matrix, combined_loss

device    = torch.device(CONFIG['device'])
_amp_dtype = torch.bfloat16 if CONFIG.get('use_bf16') else torch.float16


# ── Early stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when F1 has not improved by `min_delta` for `patience` epochs."""

    def __init__(self, patience: int = None, min_delta: float = None):
        self.patience  = patience  or CONFIG['patience']
        self.min_delta = min_delta or CONFIG['min_delta']
        self.counter   = 0
        self.best_f1   = 0.0

    def __call__(self, val_f1: float) -> bool:
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
            return False       # keep training
        self.counter += 1
        return self.counter >= self.patience   # True → stop


# ── Checkpointing ──────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    metrics: dict,
    best_f1: float,
    is_best: bool = False,
    checkpoint_dir: str = None,
) -> None:
    """Save latest (and optionally best) checkpoint."""
    ckpt_dir = checkpoint_dir or CONFIG['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    raw_model = getattr(model, '_orig_mod', model)
    state = {
        'epoch':                epoch,
        'model_state_dict':     raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'metrics':              metrics,
        'best_f1':              best_f1,
        'config':               CONFIG,
    }
    torch.save(state, os.path.join(ckpt_dir, 'latest.pth'))
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, 'best.pth'))
        print(f"  ★ Best checkpoint saved (F1={best_f1:.4f})")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scaler,
    checkpoint_dir: str = None,
) -> tuple:
    """
    Load latest checkpoint and return (start_epoch, best_f1).
    Returns (0, 0.0) if no checkpoint exists.
    """
    ckpt_dir = checkpoint_dir or CONFIG['checkpoint_dir']
    path = os.path.join(ckpt_dir, 'latest.pth')
    if not os.path.exists(path):
        print("[train] No checkpoint found – starting fresh.")
        return 0, 0.0

    ckpt = torch.load(path, map_location=device)
    raw  = getattr(model, '_orig_mod', model)
    raw.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    print(f"[train] Resumed from epoch {ckpt['epoch'] + 1} "
          f"| best F1={ckpt['best_f1']:.4f}")
    return ckpt['epoch'] + 1, ckpt['best_f1']


# ── Confusion matrix plot ──────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, epoch: int, save_dir: str) -> None:
    TN, FP, FN, TP = cm.ravel()
    labels = [[f'TN\n{TN:,}', f'FP\n{FP:,}'], [f'FN\n{FN:,}', f'TP\n{TP:,}']]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', linewidths=.5,
                xticklabels=['No Change', 'Change'],
                yticklabels=['No Change', 'Change'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Validation CM – Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch:04d}.png'), dpi=100)
    plt.close()
    print(f"  CM  TN={TN:,}  FP={FP:,}  FN={FN:,}  TP={TP:,}")


# ── Single epoch ───────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, scaler, scheduler, epoch) -> float:
    model.train()
    total = 0.0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True,
                           memory_format=torch.channels_last)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
            loss = combined_loss(model(images), masks, epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()
        if i % 20 == 0:
            print(f"  [{i:>4}/{len(loader)}] loss={loss.item():.4f}", end='\r')

    return total / len(loader)


def _validate(model, loader, epoch) -> tuple:
    model.eval()
    total = 0.0
    preds_list, tgts_list = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True,
                               memory_format=torch.channels_last)
            masks  = masks.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
                logits = model(images)
                total += combined_loss(logits, masks, epoch).item()
            preds_list.append(torch.sigmoid(logits).float().cpu().numpy())
            tgts_list.append(masks.float().cpu().numpy())

    all_preds   = np.concatenate(preds_list)
    all_targets = np.concatenate(tgts_list)
    metrics = compute_metrics(all_preds, all_targets)
    cm      = compute_confusion_matrix(all_preds, all_targets)
    return total / len(loader), metrics, cm


# ── Main training function ─────────────────────────────────────────────────

def train(
    train_loader,
    val_loader,
    resume: bool = False,
    checkpoint_dir: str = None,
) -> float:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        train_loader:   DataLoader for the training split.
        val_loader:     DataLoader for the validation split.
        resume:         If True, load latest.pth and continue.
        checkpoint_dir: Override CONFIG checkpoint_dir.

    Returns:
        Best validation F1 achieved.
    """
    ckpt_dir = checkpoint_dir or CONFIG['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    model     = get_model()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        epochs=CONFIG['max_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )
    scaler  = torch.amp.GradScaler(device='cuda',
                                   enabled=not CONFIG.get('use_bf16', True))
    stopper = EarlyStopping()

    start_epoch = best_f1 = 0
    if resume:
        start_epoch, best_f1 = load_checkpoint(model, optimizer, scaler, ckpt_dir)

    log      = []
    log_path = os.path.join(ckpt_dir, 'training_log.json')

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'F1':>7} {'IoU':>7} {'Prec':>7} {'Rec':>7} "
          f"{'LR':>10} {'Time':>6}")
    print("─" * 80)

    for epoch in range(start_epoch, CONFIG['max_epochs']):
        t0 = time.time()
        tr_loss              = _train_epoch(model, train_loader, optimizer,
                                            scaler, scheduler, epoch)
        va_loss, metrics, cm = _validate(model, val_loader, epoch)
        elapsed              = (time.time() - t0) / 60

        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{metrics['f1']:>7.4f} {metrics['iou']:>7.4f} "
              f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
              f"{lr:>10.2e} {elapsed:>5.1f}m")

        is_best = metrics['f1'] > best_f1
        if is_best or epoch % 5 == 0:
            _plot_confusion_matrix(cm, epoch, ckpt_dir)

        log.append({
            'epoch': epoch, 'train_loss': round(tr_loss, 4),
            'val_loss': round(va_loss, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
            'confusion_matrix': cm.tolist(), 'lr': lr,
            'epoch_time_min': round(elapsed, 2),
        })
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        if is_best:
            best_f1 = metrics['f1']
        save_checkpoint(epoch, model, optimizer, scaler,
                        metrics, best_f1, is_best, ckpt_dir)

        if stopper(metrics['f1']):
            print(f"\nEarly stop at epoch {epoch + 1} | best F1={best_f1:.4f}")
            break

    print(f"\n✓ Training done. Best F1={best_f1:.4f}")
    return best_f1
