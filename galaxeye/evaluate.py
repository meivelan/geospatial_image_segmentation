"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrices, qualitative visualisation,
and a full evaluation pipeline.

Usage
-----
    from galaxeye.evaluate import run_evaluation

    results = run_evaluation(
        checkpoint_path='best.pth',
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir='/results',
        threshold=0.4,
    )
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import warnings
from rasterio.errors import NotGeoreferencedWarning

from .config import CONFIG
from .model  import load_model_from_checkpoint, compute_metrics

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

device = torch.device(CONFIG['device'])
_amp_dtype = torch.bfloat16 if CONFIG.get('use_bf16') else torch.float16


# ── Inference on a split ───────────────────────────────────────────────────

def predict_split(model, loader, threshold: float = None) -> tuple:
    """
    Run inference over a DataLoader and collect predictions + targets.

    Args:
        model:     Trained model (eval mode).
        loader:    DataLoader for the split.
        threshold: Binarisation threshold.

    Returns:
        (all_preds, all_targets) — numpy arrays (N, 1, H, W).
    """
    thr = threshold if threshold is not None else CONFIG['threshold']
    model.eval()
    preds_list, tgts_list = [], []

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
                logits = model(images)
            preds_list.append(torch.sigmoid(logits).float().cpu().numpy())
            tgts_list.append(masks.numpy())
            if i % 10 == 0:
                print(f"  batch {i}/{len(loader)}", end='\r')

    print()
    return np.concatenate(preds_list), np.concatenate(tgts_list)


# ── Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    threshold: float,
    split_name: str,
    output_dir: str,
) -> None:
    """Plot, print, and save a confusion matrix for the given split."""
    preds = (all_preds > threshold).astype(int).flatten()
    tgts  = all_targets.astype(int).flatten()
    cm    = confusion_matrix(tgts, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['No-Change', 'Change']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {split_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[eval] Saved confusion matrix → {path}")

    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


# ── Qualitative visualisation ──────────────────────────────────────────────

def visualize_predictions(
    model,
    loader,
    split_name: str,
    threshold: float,
    output_dir: str,
    n_success: int = 3,
    n_failure: int = 2,
) -> None:
    """
    Plot a 5-row grid of EO / SAR / ground-truth / prediction quads,
    mixing high-IoU successes and low-IoU failures.
    """
    model.eval()
    successes, failures = [], []

    with torch.no_grad():
        for images, masks in loader:
            imgs_gpu = images.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
                preds = torch.sigmoid(model(imgs_gpu))

            preds_np  = preds.cpu().numpy()
            masks_np  = masks.numpy()
            images_np = images.numpy()

            for i in range(len(images_np)):
                pb  = (preds_np[i, 0] > threshold).astype(int)
                tgt = masks_np[i, 0].astype(int)
                if tgt.sum() == 0:
                    continue

                inter = ((pb == 1) & (tgt == 1)).sum()
                union = ((pb == 1) | (tgt == 1)).sum()
                iou   = inter / (union + 1e-6)

                sample = {'image': images_np[i], 'pred': preds_np[i, 0],
                          'pred_bin': pb, 'target': tgt, 'iou': iou}
                (successes if iou > 0.3 else failures).append(sample)

            if len(successes) >= n_success and len(failures) >= n_failure:
                break

    selected = successes[:n_success] + failures[:n_failure]
    labels   = ['Success'] * n_success + ['Failure'] * n_failure
    n_rows   = len(selected)

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis]

    for ax, title in zip(axes[0], ['EO (pre)', 'SAR (post)', 'Ground Truth', 'Prediction']):
        ax.set_title(title, fontsize=12, fontweight='bold')

    for i, (s, lbl) in enumerate(zip(selected, labels)):
        img = s['image']
        eo  = img[:3].transpose(1, 2, 0)
        eo  = (eo - eo.min()) / (eo.max() - eo.min() + 1e-6)
        sar = img[3]
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)

        axes[i, 0].imshow(eo)
        axes[i, 1].imshow(sar, cmap='gray')
        axes[i, 2].imshow(s['target'],   cmap='Reds', vmin=0, vmax=1)
        axes[i, 3].imshow(s['pred_bin'], cmap='Reds', vmin=0, vmax=1)

        colour = 'green' if lbl == 'Success' else 'red'
        axes[i, 0].set_ylabel(f'[{lbl}]\nIoU={s["iou"]:.3f}',
                               fontsize=10, color=colour, fontweight='bold')
        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle(f'Qualitative Results — {split_name}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, f'predictions_{split_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[eval] Saved qualitative plot → {path}")


# ── Results table ──────────────────────────────────────────────────────────

def print_results_table(val_metrics: dict, test_metrics: dict) -> None:
    print("\n" + "=" * 55)
    print(f"{'Metric':<15} {'Validation':>15} {'Test':>15}")
    print("=" * 55)
    for key in ['f1', 'iou', 'precision', 'recall']:
        print(f"{key.upper():<15} "
              f"{val_metrics[key]:>15.4f} "
              f"{test_metrics[key]:>15.4f}")
    print("=" * 55)


# ── Full evaluation pipeline ───────────────────────────────────────────────

def run_evaluation(
    val_loader,
    test_loader,
    checkpoint_path: str = None,
    threshold: float = None,
    output_dir: str = None,
    model=None,
) -> dict:
    """
    Load the best model and evaluate on val and test splits.

    Args:
        val_loader:       Validation DataLoader.
        test_loader:      Test DataLoader.
        checkpoint_path:  Path to best.pth (default: CONFIG checkpoint_dir/best.pth).
        threshold:        Binarisation threshold (default: CONFIG threshold).
        output_dir:       Where to save plots and results.json.
        model:            If provided, skip loading from checkpoint.

    Returns:
        dict with 'validation' and 'test' metrics.
    """
    thr      = threshold  if threshold  is not None else CONFIG['threshold']
    out_dir  = output_dir if output_dir is not None else CONFIG['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    if model is None:
        ckpt_path = checkpoint_path or os.path.join(
            CONFIG['checkpoint_dir'], 'best.pth')
        model, _ = load_model_from_checkpoint(ckpt_path)

    print(f"\n[eval] Threshold = {thr}")

    # ── Validation ──────────────────────────────────────────────────────────
    print("[eval] Evaluating val split…")
    val_preds, val_tgts = predict_split(model, val_loader, thr)
    val_metrics         = compute_metrics(val_preds, val_tgts, thr)
    print(f"  F1={val_metrics['f1']:.4f}  IoU={val_metrics['iou']:.4f}  "
          f"Prec={val_metrics['precision']:.4f}  Rec={val_metrics['recall']:.4f}")

    # ── Test ────────────────────────────────────────────────────────────────
    print("[eval] Evaluating test split…")
    tst_preds, tst_tgts = predict_split(model, test_loader, thr)
    tst_metrics         = compute_metrics(tst_preds, tst_tgts, thr)
    print(f"  F1={tst_metrics['f1']:.4f}  IoU={tst_metrics['iou']:.4f}  "
          f"Prec={tst_metrics['precision']:.4f}  Rec={tst_metrics['recall']:.4f}")

    # ── Table + plots ────────────────────────────────────────────────────────
    print_results_table(val_metrics, tst_metrics)

    plot_confusion_matrix(val_preds, val_tgts, thr, 'val',  out_dir)
    plot_confusion_matrix(tst_preds, tst_tgts, thr, 'test', out_dir)

    visualize_predictions(model, val_loader,  'val',  thr, out_dir)
    visualize_predictions(model, test_loader, 'test', thr, out_dir)

    # ── Save JSON results ────────────────────────────────────────────────────
    results = {
        'threshold':  thr,
        'validation': {k: round(v, 4) for k, v in val_metrics.items()},
        'test':       {k: round(v, 4) for k, v in tst_metrics.items()},
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved → {json_path}")
    print(json.dumps(results, indent=2))

    return results
