import os

CONFIG = {
    "dataset_root": "sample_dataset",
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

CONFIG["checkpoint_dir"] = os.path.join(
    os.path.pardir(CONFIG["dataset_root"]),
    "results",
    f"checkpoints_{CONFIG['encoder']}_{CONFIG['image_size']}px_bs{CONFIG['batch_size']}_run{len(os.listdir(os.path.join(os.path.pardir(CONFIG['dataset_root']), "results")))+1}",
)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
