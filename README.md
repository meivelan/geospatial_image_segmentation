# galaxeye – Change Detection Library

Modular Python library for the GalaxEye EO+SAR satellite change-detection pipeline.

## Package layout

```
galaxeye_lib/
├── galaxeye/
│   ├── __init__.py
│   ├── config.py          # CONFIG dict – single source of truth
│   ├── preprocessing.py   # relabelling, diff images, norm stats, pos_weight
│   ├── dataset.py         # ChangeDetectionDataset + albumentations transforms
│   ├── model.py           # U-Net, losses, metrics
│   ├── train.py           # training loop, early stopping, checkpointing
│   ├── evaluate.py        # metrics, confusion matrix, qualitative plots
│   └── pipeline.py        # one-call API
└── INTERVIEW_CHEATSHEET.ipynb
```

## one-liner

They give you a blind dataset folder:

```python
from galaxeye.pipeline import run_on_dataset

results = run_on_dataset(
    dataset_root    = '/content/blind_dataset',
    checkpoint_path = '/content/drive/MyDrive/galaxeye_checkpoints/best.pth',
    output_dir      = '/content/drive/MyDrive/galaxeye_results',
    threshold       = 0.4,
    preprocess      = True,   # False if already preprocessed
)
```

That's it. Confusion matrices, qualitative plots, and a `results.json` are
saved to `output_dir`.

## Setup in Colab

```python
!cp -r '/content/drive/MyDrive/galaxeye_lib' '/content/'
import sys; sys.path.insert(0, '/content/galaxeye_lib')
```

## Dependencies

```
segmentation-models-pytorch
rasterio
albumentations
torch >= 2.0
scikit-learn
matplotlib
seaborn
```
