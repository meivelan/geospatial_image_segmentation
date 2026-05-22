"""
model.py
--------
Model factory, combined loss function, and pixel-level metrics.

Usage
-----
    from galaxeye.model import get_model, compute_metrics, combined_loss
    model = get_model()
"""

import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from .config import CONFIG

device = torch.device(CONFIG['device'])


# ── Model factory ──────────────────────────────────────────────────────────

def get_model(
    encoder: str = None,
    encoder_weights: str = None,
    in_channels: int = None,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a U-Net with the specified encoder.

    Args:
        encoder:         timm/SMP encoder name (default from CONFIG).
        encoder_weights: 'imagenet' or None (default from CONFIG).
        in_channels:     Number of input channels (default from CONFIG).
        pretrained:      If False, forces encoder_weights=None.

    Returns:
        Model on the configured device, optionally compiled with torch.compile.
    """
    enc = encoder        or CONFIG['encoder']
    w   = encoder_weights or (CONFIG['encoder_weights'] if pretrained else None)
    c   = in_channels    or CONFIG['in_channels']

    model = smp.Unet(
        encoder_name=enc,
        encoder_weights=w,
        in_channels=c,
        classes=1,
        activation=None,
    ).to(device).to(memory_format=torch.channels_last)

    if CONFIG['compile_model'] and hasattr(torch, 'compile'):
        print("[model] torch.compile …")
        model = torch.compile(model)

    print(f"[model] U-Net / {enc} | in_channels={c} | weights={w}")
    return model


def load_model_from_checkpoint(path: str) -> tuple:
    """
    Load a model from a saved checkpoint file.

    Args:
        path: Path to the .pth checkpoint file.

    Returns:
        (model, checkpoint_dict)
    """
    ckpt   = torch.load(path, map_location=device)
    config = ckpt.get('config', CONFIG)

    model = smp.Unet(
        encoder_name=config['encoder'],
        encoder_weights=None,          # don't re-download
        in_channels=config['in_channels'],
        classes=1,
        activation=None,
    ).to(device)

    raw = getattr(model, '_orig_mod', model)
    raw.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"[model] Loaded from {path}")
    print(f"        epoch={ckpt.get('epoch', '?')} | "
          f"best_F1={ckpt.get('best_f1', '?'):.4f}")
    return model, ckpt


# ── Loss functions ─────────────────────────────────────────────────────────

_dice_loss = smp.losses.DiceLoss(mode='binary').to(device)


def _get_bce(pos_weight: float = None) -> nn.Module:
    pw = pos_weight or CONFIG['pos_weight']
    pw_tensor = torch.tensor([pw], device=device)
    return smp.losses.SoftBCEWithLogitsLoss(pos_weight=pw_tensor).to(device)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epoch: int = 999,
    pos_weight: float = None,
) -> torch.Tensor:
    """
    Dice + gradually introduced weighted BCE loss.

    For the first 5 epochs only Dice is used (stabilises early training).
    BCE weight linearly ramps from 0 to 1 over epochs 5-15.

    Args:
        pred:       Raw logits from the model (before sigmoid).
        target:     Ground-truth mask tensor.
        epoch:      Current training epoch (controls BCE ramp).
        pos_weight: Override positive weight for BCE.

    Returns:
        Scalar loss tensor.
    """
    t = target.to(pred.device)
    if epoch < 5:
        return _dice_loss(pred, t)
    bce_w = min(1.0, (epoch - 5) / 10.0)
    return _dice_loss(pred, t) + bce_w * _get_bce(pos_weight)(pred, t)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Compute pixel-level F1, IoU, Precision, and Recall.

    Args:
        all_preds:   Sigmoid probabilities, shape (N, 1, H, W).
        all_targets: Ground-truth masks,    shape (N, 1, H, W).
        threshold:   Binarisation threshold (default from CONFIG).

    Returns:
        Dict with keys: f1, iou, precision, recall.
    """
    thr   = threshold if threshold is not None else CONFIG['threshold']
    preds = (all_preds > thr).astype(int).flatten()
    tgts  = all_targets.astype(int).flatten()

    inter = ((preds == 1) & (tgts == 1)).sum()
    union = ((preds == 1) | (tgts == 1)).sum()

    return {
        'f1':        float(f1_score(tgts, preds, zero_division=0)),
        'iou':       float(inter) / (float(union) + 1e-6),
        'precision': float(precision_score(tgts, preds, zero_division=0)),
        'recall':    float(recall_score(tgts, preds, zero_division=0)),
    }


def compute_confusion_matrix(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float = None,
) -> np.ndarray:
    """
    Return 2×2 pixel-level confusion matrix [[TN, FP], [FN, TP]].
    """
    thr   = threshold if threshold is not None else CONFIG['threshold']
    preds = (all_preds > thr).astype(int).flatten()
    tgts  = all_targets.astype(int).flatten()
    return confusion_matrix(tgts, preds, labels=[0, 1])
