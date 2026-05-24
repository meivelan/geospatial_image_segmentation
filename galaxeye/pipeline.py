"""
pipeline.py  –  one-call wrappers for the interview.

All the logic is in the other modules using your exact notebook code.
This file just chains them together.
"""

import os
import json
import torch

from .config import CONFIG
from .preprocessing import (
    re_labeling,
    compute_and_save_diff,
    valid__files,
    filter_corrupted_valid_files,
    compute_norm_stats,
    compute_pos_weight_for_valid,
    check_corrupted_files,
)
from .dataset import ChangeDetectionDataset, get_loaders, get_test_loader, _loader_kw
from .model import get_model
from .train import train
from .evaluate import test_model, plot_test_confusion_matrix, visualize_predictions


def run_preprocessing(dataset_root=None, test_split=False):
    """Steps 1-7 from the preprocessing notebook in order."""
    r = dataset_root or CONFIG["dataset_root"]

    if not test_split:
        tr = r + "/train/"
        va = r + "/val/"
        re_labeling(tr + "target")
        re_labeling(va + "target")

        for split_path in [tr, va]:
            compute_and_save_diff(split_path)

        valid__files(tr, r + "/")
        filter_corrupted_valid_files(r)
        compute_norm_stats(r)
        compute_pos_weight_for_valid(tr, r + "/")

        check_corrupted_files(r, "train", include_diff=True)
        check_corrupted_files(r, "val", include_diff=True)
    else:
        print("Preprocessing only test split")
    te = r + "/test/"
    re_labeling(te + "target")
    compute_and_save_diff(te)

    check_corrupted_files(r, "test", include_diff=True)

    print("\nPreprocessing complete.")


def run_training(dataset_root=None):
    """Build loaders → train → return best_f1."""
    root = dataset_root or CONFIG["dataset_root"]

    train_loader, val_loader = get_loaders(root)
    best_f1 = train(train_loader, val_loader)
    return best_f1


def run_evaluation(
    dataset_root=None, checkpoint_path=None, test_data_root=None, num_samples=5
):
    """Load best.pth → test_model → visualize_predictions."""
    root = dataset_root or CONFIG["dataset_root"]
    ckpt_path = checkpoint_path or os.path.join(CONFIG["checkpoint_dir"], "best.pth")
    test_root = test_data_root or "/content/"

    with open(root + "/norm_stats.json") as f:
        norm_stats = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "config" in ckpt:
        CONFIG.update(ckpt["config"])

    model = get_model()
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader, test_dataset = get_test_loader(
        dataset_root=test_root,
        norm_stats=norm_stats,
        image_size=CONFIG["image_size"],
    )

    test_loss, test_metrics, test_cm = test_model(model, test_loader)

    run_name = os.path.basename(os.path.dirname(ckpt_path))
    results_dir = os.path.join(os.path.dirname(ckpt_path), "test_results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nTest Loss:      {test_loss:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")
    print(f"Test IoU:       {test_metrics['iou']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")

    plot_test_confusion_matrix(test_cm, run_name, results_dir)
    visualize_predictions(model, test_dataset, num_samples=num_samples)

    return test_loss, test_metrics, test_cm


def run_on_dataset(
    dataset_root, checkpoint_path=None, test_data_root=None, num_samples=5
):
    run_preprocessing(dataset_root, test_split=True)
    if not checkpoint_path:
        print("Checkpoint Path required...")
        return
    return run_evaluation(
        dataset_root=dataset_root or CONFIG['dataset_root'],
        checkpoint_path=checkpoint_path,
        test_data_root=test_data_root or dataset_root,
        num_samples=num_samples,
    )


def run_full_pipeline(dataset_root=None):
    """preprocess → train → evaluate"""
    run_preprocessing(dataset_root)
    run_training(dataset_root)
    run_evaluation(dataset_root)
