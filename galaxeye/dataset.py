"""
dataset.py
----------
PyTorch Dataset for the GalaxEye EO+SAR change-detection task.

Each sample returns a 5-channel image tensor (3 EO + 1 SAR + 1 diff)
and a binary mask (1 = change, 0 = no-change).

Usage
-----
    from galaxeye.dataset import ChangeDetectionDataset, get_loaders

    train_loader, val_loader, test_loader = get_loaders(
        dataset_root='/path/to/data',
        norm_stats=norm_stats,
        valid_files=valid_files,
    )
"""

import os
import json
import warnings
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rasterio.errors import NotGeoreferencedWarning

from .config import CONFIG

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


# ── Augmentation transforms ────────────────────────────────────────────────

def get_transforms(image_size: int, is_train: bool, norm_stats: dict) -> A.Compose:
    """
    Build an albumentations pipeline for the given split.

    Args:
        image_size: Target H x W for resizing.
        is_train:   If True, applies geometric augmentations.
        norm_stats: Dict with eo_mean, eo_std, sar_mean, sar_std,
                    diff_mean, diff_std (from preprocessing).

    Returns:
        albumentations.Compose pipeline.
    """
    mean = norm_stats['eo_mean'] + [norm_stats['sar_mean']] + [norm_stats['diff_mean']]
    std  = norm_stats['eo_std']  + [norm_stats['sar_std']]  + [norm_stats['diff_std']]

    common = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    augment = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ] if is_train else []

    return A.Compose(augment + common)


# ── Dataset ────────────────────────────────────────────────────────────────

class ChangeDetectionDataset(Dataset):
    """
    EO + SAR change-detection dataset.

    Directory layout expected at `root_dir/<split>/`:
        pre-event/          - 3-band optical GeoTIFF (uint8)
        post-event/         - 1-band SAR GeoTIFF (uint8)
        diff/               - 1-band |EO_gray - SAR| GeoTIFF (float32)
        re_labelled-target/ - 1-band binary mask GeoTIFF (uint8, 0/1)

    Args:
        root_dir:    Root directory of the dataset.
        split:       'train', 'val', or 'test'.
        norm_stats:  Normalisation statistics dict.
        image_size:  Resize target (default from CONFIG).
        is_train:    Enable augmentations.
        valid_files: If given (train split), restricts to these filenames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        norm_stats: dict,
        image_size: int = None,
        is_train: bool = False,
        valid_files: list = None,
    ):
        self.root_dir   = root_dir
        self.split      = split
        self.is_train   = is_train
        self.image_size = image_size or CONFIG['image_size']

        self.pre_dir    = os.path.join(root_dir, split, 'pre-event')
        self.post_dir   = os.path.join(root_dir, split, 'post-event')
        self.diff_dir   = os.path.join(root_dir, split, 'diff')
        self.target_dir = os.path.join(root_dir, split, 're_labelled-target')

        self.transform = get_transforms(self.image_size, is_train, norm_stats)

        if is_train and valid_files is not None:
            self.filenames = list(valid_files)
        else:
            self.filenames = sorted(
                f for f in os.listdir(self.pre_dir)
                if f.endswith(('.tif', '.tiff'))
            )

        print(f"[Dataset] {split:5s}: {len(self.filenames)} files")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        try:
            # EO: 3-band optical, normalise to [0, 1]
            with rasterio.open(os.path.join(self.pre_dir, fname)) as src:
                eo = src.read().astype(np.float32) / 255.0          # (3, H, W)

            # SAR: single-band, normalise to [0, 1]
            with rasterio.open(os.path.join(self.post_dir, fname)) as src:
                sar = src.read(1).astype(np.float32) / 255.0        # (H, W)

            # Diff: already float32
            with rasterio.open(os.path.join(self.diff_dir, fname)) as src:
                diff = src.read(1).astype(np.float32)               # (H, W)

            # Mask
            with rasterio.open(os.path.join(self.target_dir, fname)) as src:
                mask = src.read(1).astype(np.float32)               # (H, W)

            # Stack into HWC for albumentations
            image = np.concatenate([
                eo.transpose(1, 2, 0),          # HW3
                sar[:, :, np.newaxis],           # HW1
                diff[:, :, np.newaxis],          # HW1
            ], axis=-1)                          # HW5

            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']            # (5, H, W) tensor
            mask  = transformed['mask'].unsqueeze(0).float()   # (1, H, W)

            return image, mask

        except Exception as e:
            print(f"[Dataset] Error loading {fname}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# ── DataLoader factory ─────────────────────────────────────────────────────

def get_loaders(
    dataset_root: str,
    norm_stats: dict,
    valid_files: list = None,
    image_size: int = None,
    batch_size: int = None,
    num_workers: int = None,
) -> tuple:
    """
    Build train / val / test DataLoaders.

    Args:
        dataset_root: Root directory of the dataset.
        norm_stats:   Normalisation statistics dict (from preprocessing).
        valid_files:  Training filenames to include (non-empty tiles).
                      If None, reads valid_train_files.txt from dataset_root.
        image_size:   Override CONFIG image_size.
        batch_size:   Override CONFIG batch_size.
        num_workers:  Override CONFIG num_workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if valid_files is None:
        txt = os.path.join(dataset_root, 'valid_train_files.txt')
        with open(txt) as f:
            valid_files = f.read().splitlines()

    sz  = image_size  or CONFIG['image_size']
    bs  = batch_size  or CONFIG['batch_size']
    nw  = num_workers or CONFIG['num_workers']

    train_ds = ChangeDetectionDataset(
        dataset_root, 'train', norm_stats, sz, is_train=True,
        valid_files=valid_files,
    )
    val_ds = ChangeDetectionDataset(
        dataset_root, 'val', norm_stats, sz, is_train=False,
    )
    test_ds = ChangeDetectionDataset(
        dataset_root, 'test', norm_stats, sz, is_train=False,
    )

    loader_kw = dict(
        batch_size=bs, num_workers=nw,
        pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=4,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"[Loaders] train={len(train_loader)} | "
          f"val={len(val_loader)} | test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ── Convenience: load loaders from saved stats ─────────────────────────────

def get_loaders_from_disk(dataset_root: str, **kwargs) -> tuple:
    """
    Same as get_loaders() but reads norm_stats.json and
    valid_train_files.txt automatically from dataset_root.
    """
    stats_path = os.path.join(dataset_root, 'norm_stats.json')
    with open(stats_path) as f:
        norm_stats = json.load(f)

    return get_loaders(dataset_root, norm_stats, **kwargs)
