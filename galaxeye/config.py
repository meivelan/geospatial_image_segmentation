import os
from pathlib import Path

CONFIG = {
    "dataset_root": Path.cwd(),
    "image_size": 512,
    "batch_size": 8,
    "num_workers": 8,
    "lr": 3e-5,
    "max_epochs": 200,
    "patience": 15,
    "min_delta": 0.001,
    "pos_weight": 15,
    "threshold": 0.3,
    "seed": 42,
    "encoder": "efficientnet-b0",
    "encoder_weights": "imagenet",
    "in_channels": 5,
    "use_bf16": True,
    "compile_model": True,
    "grad_clip": 1.0,
}

run = len(os.listdir(os.path.join(CONFIG["dataset_root"], "results"))) + 1
CONFIG["checkpoint_dir"] = os.path.join(
    CONFIG["dataset_root"],
    "results",
    f"checkpoints_{CONFIG['encoder']}_{CONFIG['image_size']}px_bs{CONFIG['batch_size']}_run{run}",
)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)


def get_config():
    return CONFIG


def set_config(
    dataset_root=None,
    image_size=None,
    batch_size=None,
    num_workers=None,
    lr=None,
    max_epochs=None,
    patience=None,
    min_delta=None,
    pos_weight=None,
    threshold=None,
    seed=None,
    encoder=None,
    encoder_weights=None,
    in_channels=None,
    use_bf16=None,
    compile_model=None,
    grad_clip=None
):
    if dataset_root:
        CONFIG['dataset_root'] = dataset_root
    if image_size:
        CONFIG['image_size'] = image_size
    if batch_size:
        CONFIG['batch_size'] = batch_size
    if num_workers:
        CONFIG['num_workers'] = num_workers
    if lr:
        CONFIG['lr'] = lr
    if max_epochs:
        CONFIG['max_epochs'] = max_epochs
    if patience:
        CONFIG['patience'] = patience
    if min_delta:
        CONFIG['min_delta'] = min_delta
    if pos_weight:
        CONFIG['pos_weight'] = pos_weight
    if threshold:
        CONFIG['threshold'] = threshold
    if seed:
        CONFIG['seed'] = seed
    if encoder:
        CONFIG['encoder'] = encoder
    if encoder_weights:
        CONFIG['encoder_weights'] = encoder_weights
    if in_channels:
        CONFIG['in_channels'] = in_channels
    if use_bf16:
        CONFIG['use_bf16'] = use_bf16
    if compile_model:
        CONFIG['compile_model'] = compile_model
    if grad_clip:
        CONFIG['grad_clip'] = grad_clip