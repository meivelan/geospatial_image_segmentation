"""
GalaxEye Change Detection Library
==================================
A modular library for satellite change detection (EO + SAR fusion).

Quick start
-----------
    from galaxeye.pipeline import run_on_dataset
    run_on_dataset("/path/to/dataset", checkpoint_path="best.pth")

Modules
-------
    config        - default hyper-parameters (CONFIG dict)
    preprocessing - relabelling, diff images, norm stats, pos_weight
    dataset       - ChangeDetectionDataset + albumentations transforms
    model         - U-Net factory, loss functions, metrics
    train         - training loop, early stopping, checkpointing
    evaluate      - metrics, confusion matrix, qualitative visualisation
    pipeline      - one-call API: preprocess → train → evaluate
"""

from .config import CONFIG          # noqa: F401
