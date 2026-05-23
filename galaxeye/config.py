import os

CONFIG = {
    'dataset_root':    '/content/galaxeye_assessment',
    'image_size':      512,
    'batch_size':      8,
    'num_workers':     8,
    'lr':              3e-5,
    'max_epochs':      200,
    'patience':        15,
    'min_delta':       0.001,
    'pos_weight':      15,
    'threshold':       0.3,
    'seed':            42,
    'encoder':         'efficientnet-b0',
    'encoder_weights': 'imagenet',
    'in_channels':     5,
    'use_bf16':        True,
    'compile_model':   True,
    'grad_clip':       1.0,
}

CONFIG['checkpoint_dir'] = os.path.join(
    '/content/drive/MyDrive/galaxeye_assessment/',
    f"checkpoints_{CONFIG['encoder']}_{CONFIG['image_size']}px_bs{CONFIG['batch_size']}_v4"
)
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
