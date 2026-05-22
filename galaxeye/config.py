"""
config.py
---------
Single source of truth for every hyper-parameter used across the pipeline.
Override individual keys before importing other modules:

    from galaxeye.config import CONFIG
    CONFIG['threshold'] = 0.35
"""

import os
import torch

CONFIG = {
    # ── Paths ─────────────────────────────────────────────
    'dataset_root':    '/content/galaxeye_assessment',
    'checkpoint_dir':  '/content/drive/MyDrive/galaxeye_checkpoints',
    'output_dir':      '/content/drive/MyDrive/galaxeye_results',

    # ── Data ──────────────────────────────────────────────
    'image_size':      512,
    'batch_size':      8,
    'num_workers':     8,
    'in_channels':     5,          # EO(3) + SAR(1) + diff(1)

    # ── Model ─────────────────────────────────────────────
    'encoder':         'efficientnet-b0',
    'encoder_weights': 'imagenet',

    # ── Training ──────────────────────────────────────────
    'lr':              3e-5,
    'max_epochs':      200,
    'patience':        15,
    'min_delta':       0.001,
    'pos_weight':      15,
    'threshold':       0.3,
    'seed':            42,
    'grad_clip':       1.0,
    'use_bf16':        True,
    'compile_model':   True,

    # ── Runtime (auto-detected) ───────────────────────────
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
}
