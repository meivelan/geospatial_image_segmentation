import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

from .config import CONFIG


def get_transforms(image_size, is_train, norm_stats):
    mean = (norm_stats['eo_mean'] +
            [norm_stats['sar_mean']] +
            [norm_stats['diff_mean']])
    std  = (norm_stats['eo_std'] +
            [norm_stats['sar_std']] +
            [norm_stats['diff_std']])

    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split, norm_stats,
                 image_size=512, is_train=False,
                 valid_files=None):
        self.root_dir  = root_dir
        self.split     = split
        self.is_train  = is_train
        self.transform = get_transforms(image_size, is_train, norm_stats)

        self.pre_dir    = os.path.join(root_dir, split, 'pre-event')
        self.post_dir   = os.path.join(root_dir, split, 'post-event')
        self.target_dir = os.path.join(root_dir, split, 're_labelled-target')
        self.diff_dir   = os.path.join(root_dir, split, 'diff')

        if is_train and valid_files is not None:
            self.filenames = valid_files
        else:
            self.filenames = sorted([
                f for f in os.listdir(self.pre_dir)
                if f.endswith(('.tif', '.tiff'))
            ])

        print(f"{split}: {len(self.filenames)} files")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        try:
            # Load EO (3 channels) — normalize to [0,1]
            with rasterio.open(os.path.join(self.pre_dir, fname)) as src:
                eo = src.read().astype(np.float32) / 255.0

            # Load SAR (1 channel) — normalize to [0,1]
            with rasterio.open(os.path.join(self.post_dir, fname)) as src:
                sar = src.read(1).astype(np.float32) / 255.0

            # Load diff (already float32, not uint8)
            with rasterio.open(os.path.join(self.diff_dir, fname)) as src:
                diff = src.read(1).astype(np.float32)

            # Load mask
            with rasterio.open(os.path.join(self.target_dir, fname)) as src:
                mask = src.read(1).astype(np.float32)

            image = np.concatenate([
                eo.transpose(1, 2, 0),       # HW3
                sar[:, :, np.newaxis],        # HW1
                diff[:, :, np.newaxis]        # HW1
            ], axis=-1)                       # HW5

            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']      # (5, H, W) tensor
            mask  = transformed['mask']       # (H, W) tensor

            mask = mask.unsqueeze(0).float()  # (1, H, W)

            return image, mask

        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return self.__getitem__((idx + 1) % len(self))


_loader_kw = dict(
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


def get_loaders(dataset_root=None, norm_stats=None, valid_files=None, image_size=None):
    root = dataset_root or CONFIG['dataset_root']
    sz   = image_size   or CONFIG['image_size']

    if norm_stats is None:
        with open(root + '/norm_stats.json') as f:
            norm_stats = json.load(f)

    if valid_files is None:
        valid_files = open(root + '/valid_train_files.txt').read().splitlines()

    train_dataset = ChangeDetectionDataset(
        root_dir=root, split='train', norm_stats=norm_stats,
        image_size=sz, is_train=True, valid_files=valid_files
    )
    val_dataset = ChangeDetectionDataset(
        root_dir=root, split='val', norm_stats=norm_stats,
        image_size=sz, is_train=False
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **_loader_kw)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **_loader_kw)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    return train_loader, val_loader


def get_test_loader(dataset_root=None, norm_stats=None, image_size=None):
    root = dataset_root or '/content/'
    sz   = image_size   or CONFIG['image_size']

    if norm_stats is None:
        with open(CONFIG['dataset_root'] + '/norm_stats.json') as f:
            norm_stats = json.load(f)

    test_dataset = ChangeDetectionDataset(
        root_dir=root, split='test', norm_stats=norm_stats,
        image_size=sz, is_train=False
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **_loader_kw)
    print(f"Test batches : {len(test_loader)}")
    return test_loader, test_dataset
