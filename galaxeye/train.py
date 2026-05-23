import torch
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CONFIG
from .model  import (get_model, combined_loss,
                     compute_metrics, compute_confusion_matrix)

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_amp_dtype = torch.bfloat16 if CONFIG['use_bf16'] else torch.float16


def plot_confusion_matrix(cm, epoch, save_dir):
    """Plot and save a labelled confusion-matrix heatmap."""
    TN, FP, FN, TP = cm.ravel()
    labels = [
        [f'TN\n{TN:,}', f'FP\n{FP:,}'],
        [f'FN\n{FN:,}', f'TP\n{TP:,}'],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=labels, fmt='', cmap='Blues', linewidths=.5,
        xticklabels=['No Change', 'Change'],
        yticklabels=['No Change', 'Change'],
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(f'Validation Confusion Matrix – Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch:04d}.png'), dpi=120)
    plt.show()
    plt.close()
    print(f"  CM  TN={TN:,}  FP={FP:,}  FN={FN:,}  TP={TP:,}")


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_f1   = 0

    def __call__(self, val_f1):
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(epoch, model, optimizer, scaler, metrics, best_f1,
                    is_best=False):
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
    torch.save(state, os.path.join(CONFIG['checkpoint_dir'], 'latest.pth'))
    if is_best:
        torch.save(state, os.path.join(CONFIG['checkpoint_dir'], 'best.pth'))
        print(f"Best checkpoint saved (F1={best_f1:.4f})")


def load_checkpoint(model, optimizer, scaler):
    path = os.path.join(CONFIG['checkpoint_dir'], 'latest.pth')
    if not os.path.exists(path):
        print("No checkpoint found – starting fresh.")
        return 0, 0.0
    ckpt      = torch.load(path, map_location=device)
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    print(f"Resumed from epoch {ckpt['epoch']+1} | best F1={ckpt['best_f1']:.4f}")
    return ckpt['epoch'] + 1, ckpt['best_f1']


def train_epoch(model, loader, optimizer, scaler, scheduler, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True,
                           memory_format=torch.channels_last)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
            preds = model(images)
            loss  = combined_loss(preds, masks, epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"  [{batch_idx:>4}/{len(loader)}] "
                  f"loss={loss.item():.4f}", end='\r')
    return total_loss / len(loader)


def validate(model, loader, epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True,
                               memory_format=torch.channels_last)
            masks  = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
                preds = model(images)
                loss  = combined_loss(preds, masks, epoch)

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(preds).float().cpu().numpy())
            all_targets.append(masks.float().cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets, CONFIG['threshold'])
    cm      = compute_confusion_matrix(all_preds, all_targets,
                                       CONFIG['threshold'])
    return total_loss / len(loader), metrics, cm


def train(train_loader, val_loader):
    model     = get_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        epochs=CONFIG['max_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    scaler  = torch.amp.GradScaler(
        device='cuda', enabled=not CONFIG['use_bf16'])
    stopper = EarlyStopping(CONFIG['patience'], CONFIG['min_delta'])

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scaler)

    log      = []
    log_path = os.path.join(CONFIG['checkpoint_dir'], 'training_log.json')

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'F1':>7} {'IoU':>7} {'Prec':>7} {'Rec':>7} "
          f"{'LR':>10} {'Time':>6}")
    print("─" * 80)

    for epoch in range(start_epoch, CONFIG['max_epochs']):
        t0 = time.time()

        tr_loss              = train_epoch(
            model, train_loader, optimizer, scaler, scheduler, epoch)
        va_loss, metrics, cm = validate(model, val_loader, epoch)
        epoch_min            = (time.time() - t0) / 60

        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{metrics['f1']:>7.4f} {metrics['iou']:>7.4f} "
              f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
              f"{current_lr:>10.2e} {epoch_min:>5.1f}m")

        is_best = metrics['f1'] > best_f1
        if is_best or epoch % 5 == 0:
            plot_confusion_matrix(cm, epoch, CONFIG['checkpoint_dir'])

        log.append({
            'epoch':            epoch,
            'train_loss':       round(tr_loss, 4),
            'val_loss':         round(va_loss, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
            'confusion_matrix': cm.tolist(),
            'lr':               current_lr,
            'epoch_time_min':   round(epoch_min, 2),
        })
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        if is_best:
            best_f1 = metrics['f1']
        save_checkpoint(epoch, model, optimizer, scaler,
                        metrics, best_f1, is_best)

        if stopper(metrics['f1']):
            print(f"\nEarly stop epoch {epoch+1} | F1={best_f1:.4f}")
            break

    print(f"\nDone. Best F1={best_f1:.4f}")
    return best_f1
