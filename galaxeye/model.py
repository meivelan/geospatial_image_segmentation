import torch
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from .config import CONFIG

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_amp_dtype = torch.bfloat16 if CONFIG['use_bf16'] else torch.float16


def get_model():
    model = smp.Unet(
        encoder_name=CONFIG['encoder'],
        encoder_weights=CONFIG['encoder_weights'],
        in_channels=CONFIG['in_channels'],
        classes=1,
        activation=None,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)
    if CONFIG['compile_model'] and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile …")
        model = torch.compile(model)
    return model


# ── Loss ──────────────────────────────────────────────────────────────────
dice_loss         = smp.losses.DiceLoss(mode='binary').to(device)
pos_weight_tensor = torch.tensor([CONFIG['pos_weight']], device=device)
bce_loss          = smp.losses.SoftBCEWithLogitsLoss(
    pos_weight=pos_weight_tensor).to(device)


def combined_loss(pred, target, epoch):
    t = target.to(pred.device)
    if epoch < 5:
        return dice_loss(pred, t)
    bce_weight = min(1.0, (epoch - 5) / 10.0)
    return dice_loss(pred, t) + bce_weight * bce_loss(pred, t)


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_targets, threshold):
    preds_b   = (all_preds   > threshold).astype(int).flatten()
    targets_f = all_targets.astype(int).flatten()
    inter = ((preds_b == 1) & (targets_f == 1)).sum()
    union = ((preds_b == 1) | (targets_f == 1)).sum()
    return {
        'f1':        f1_score(targets_f, preds_b, zero_division=0),
        'iou':       float(inter) / (float(union) + 1e-6),
        'precision': precision_score(targets_f, preds_b, zero_division=0),
        'recall':    recall_score(targets_f, preds_b, zero_division=0),
    }


def compute_confusion_matrix(all_preds, all_targets, threshold):
    """Return pixel-level confusion matrix [[TN, FP], [FN, TP]]."""
    preds_b   = (all_preds   > threshold).astype(int).flatten()
    targets_f = all_targets.astype(int).flatten()
    return confusion_matrix(targets_f, preds_b, labels=[0, 1])
