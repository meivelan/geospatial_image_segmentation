"""
pipeline.py
-----------
High-level one-call API designed for the interview scenario:

    "Here's a blind dataset, run your pipeline on it."

─────────────────────────────────────────────────────────────

SCENARIO 1 - Evaluate only (most likely in the interview)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
They give you a new dataset folder, you already have a trained model:

    from galaxeye.pipeline import run_on_dataset

    results = run_on_dataset(
        dataset_root   = '/path/to/blind/dataset',
        checkpoint_path= '/content/drive/MyDrive/galaxeye_checkpoints/best.pth',
        output_dir     = '/content/drive/MyDrive/galaxeye_results',
        threshold      = 0.4,          # your tuned threshold
        preprocess     = True,         # set False if diff/relabel already done
    )

─────────────────────────────────────────────────────────────

SCENARIO 2 - Full pipeline (preprocess + train + evaluate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from galaxeye.pipeline import run_full_pipeline

    run_full_pipeline(
        dataset_root  = '/path/to/dataset',
        checkpoint_dir= '/checkpoints',
        output_dir    = '/results',
    )
"""

import os
import json
import torch
from .config       import CONFIG
from .preprocessing import run_all as preprocess_all
from .dataset      import get_loaders
from .model        import load_model_from_checkpoint
from .train        import train
from .evaluate     import run_evaluation


# ── Evaluate-only (interview scenario) ────────────────────────────────────

def run_on_dataset(
    dataset_root: str,
    checkpoint_path: str = None,
    output_dir: str = None,
    threshold: float = None,
    preprocess: bool = True,
    splits_to_eval: tuple = ('val', 'test'),
    image_size: int = None,
    batch_size: int = None,
) -> dict:
    """
    Preprocess a dataset (optional), then evaluate a saved model on it.

    This is the one-liner you'll use during the interview.

    Args:
        dataset_root:    Root directory of the (blind) dataset.
        checkpoint_path: Path to best.pth. Defaults to CONFIG checkpoint_dir.
        output_dir:      Where to save plots and results.json.
        threshold:       Sigmoid threshold (default: CONFIG['threshold']).
        preprocess:      Run re-labelling, diff, norm-stats etc. first.
                         Set False if the dataset is already preprocessed.
        splits_to_eval:  Which splits to evaluate ('val' and/or 'test').
        image_size:      Override image size.
        batch_size:      Override batch size.

    Returns:
        Results dict with validation / test metrics.
    """
    print("=" * 60)
    print("GalaxEye Evaluation Pipeline")
    print(f"  dataset_root : {dataset_root}")
    print("=" * 60)

    # ── 1. Preprocessing ────────────────────────────────────────────────────
    if preprocess:
        prep_result = preprocess_all(dataset_root)
        norm_stats  = prep_result['norm_stats']
    else:
        stats_path = os.path.join(dataset_root, 'norm_stats.json')
        with open(stats_path) as f:
            norm_stats = json.load(f)
        print(f"[pipeline] Loaded existing norm_stats from {stats_path}")

    # ── 2. Build DataLoaders ─────────────────────────────────────────────────
    # For evaluation we don't need the train loader.
    # We build val and test loaders over the new dataset.
    from .dataset import ChangeDetectionDataset
    from torch.utils.data import DataLoader

    sz = image_size or CONFIG['image_size']
    bs = batch_size or CONFIG['batch_size']
    nw = CONFIG['num_workers']

    loader_kw = dict(
        batch_size=bs, num_workers=nw,
        pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=4,
    )

    loaders = {}
    for split in splits_to_eval:
        split_dir = os.path.join(dataset_root, split, 'pre-event')
        if not os.path.isdir(split_dir):
            print(f"[pipeline] Skipping '{split}' - directory not found.")
            continue
        ds = ChangeDetectionDataset(
            dataset_root, split, norm_stats, sz, is_train=False)
        loaders[split] = DataLoader(ds, shuffle=False, **loader_kw)

    if len(loaders) < 2 and 'val' not in loaders:
        # Fallback: if only test exists, duplicate as val
        if 'test' in loaders:
            loaders['val'] = loaders['test']
            print("[pipeline] No val split found - using test split for both.")
        else:
            raise RuntimeError("No usable splits found. "
                               "Check that the dataset has pre-event/, post-event/, "
                               "re_labelled-target/, and diff/ under each split.")

    val_loader  = loaders.get('val',  loaders.get('test'))
    test_loader = loaders.get('test', loaders.get('val'))

    # ── 3. Load model ─────────────────────────────────────────────────────────
    ckpt_path = checkpoint_path or os.path.join(
        CONFIG['checkpoint_dir'], 'best.pth')
    model, ckpt = load_model_from_checkpoint(ckpt_path)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    out = output_dir or CONFIG['output_dir']
    results = run_evaluation(
        val_loader=val_loader,
        test_loader=test_loader,
        threshold=threshold,
        output_dir=out,
        model=model,
    )

    print("\n✓ Pipeline complete.")
    print(f"✓ Outputs saved to: {out}")
    return results


# ── Full pipeline: preprocess + train + evaluate ───────────────────────────

def run_full_pipeline(
    dataset_root: str,
    checkpoint_dir: str = None,
    output_dir: str = None,
    threshold: float = None,
    resume: bool = False,
) -> dict:
    """
    End-to-end pipeline: preprocessing → training → evaluation.

    Args:
        dataset_root:   Root directory of the dataset.
        checkpoint_dir: Where to save model checkpoints.
        output_dir:     Where to save evaluation outputs.
        threshold:      Sigmoid threshold for evaluation.
        resume:         Resume training from latest checkpoint.

    Returns:
        Evaluation results dict.
    """
    print("=" * 60)
    print("GalaxEye Full Pipeline")
    print("=" * 60)

    # Override CONFIG paths if provided
    if checkpoint_dir:
        CONFIG['checkpoint_dir'] = checkpoint_dir
    if output_dir:
        CONFIG['output_dir'] = output_dir

    # 1. Preprocess
    prep   = preprocess_all(dataset_root)
    nstats = prep['norm_stats']
    valid  = prep['valid_files']

    # 2. Build loaders
    train_loader, val_loader, test_loader = get_loaders(
        dataset_root, norm_stats=nstats, valid_files=valid)

    # 3. Train
    best_f1 = train(train_loader, val_loader, resume=resume)

    # 4. Evaluate
    results = run_evaluation(
        val_loader=val_loader,
        test_loader=test_loader,
        threshold=threshold,
    )

    return results
