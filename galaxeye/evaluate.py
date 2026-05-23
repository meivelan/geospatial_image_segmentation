import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning

from .config import CONFIG
from .model  import (get_model, combined_loss,
                     compute_metrics, compute_confusion_matrix)

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_amp_dtype = torch.bfloat16 if CONFIG['use_bf16'] else torch.float16


def plot_test_confusion_matrix(cm, run_name, save_dir):
    """Plot and save a labelled confusion-matrix heatmap for test results."""
    TN, FP, FN, TP = cm.ravel()
    labels = [
        [f'TN\n{TN:,}', f'FP\n{FP:,}'],
        [f'FN\n{FN:,}', f'TP\n{TP:,}'],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=labels, fmt='', cmap='Blues', linewidths=.5,
        xticklabels=['No Change', 'Change'],
        yticklabels=['No Change', 'Change'],
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(f'Test Confusion Matrix for {run_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'test_cm_{run_name}.png'), dpi=120)
    plt.show()
    plt.close()
    print(f"  CM  TN={TN:,}  FP={FP:,}  FN={FN:,}  TP={TP:,}")


def test_model(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True,
                               memory_format=torch.channels_last)
            masks  = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=_amp_dtype):
                preds = model(images)
                loss  = combined_loss(preds, masks, 0)  # epoch=0 for test

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(preds).float().cpu().numpy())
            all_targets.append(masks.float().cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets, CONFIG['threshold'])
    cm      = compute_confusion_matrix(all_preds, all_targets,
                                       CONFIG['threshold'])
    return total_loss / len(loader), metrics, cm


def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    samples_processed = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if samples_processed >= num_samples:
                break

            fname = dataset.filenames[idx]

            with rasterio.open(os.path.join(dataset.root_dir, dataset.split,
                                            'pre-event', fname)) as src:
                original_eo = src.read().astype(np.float32) / 255.0
            original_eo_display = original_eo.transpose(1, 2, 0)[:, :, :3]

            with rasterio.open(os.path.join(dataset.root_dir, dataset.split,
                                            'post-event', fname)) as src:
                original_sar = src.read(1).astype(np.float32) / 255.0
            original_sar_display = original_sar

            transformed_image_tensor, true_mask_tensor = dataset[idx]

            input_image_batch = transformed_image_tensor.unsqueeze(0).to(
                device, non_blocking=True,
                memory_format=torch.channels_last)

            preds        = model(input_image_batch)
            preds_binary = (torch.sigmoid(preds) > CONFIG['threshold']).float()

            input_image_for_display = transformed_image_tensor.cpu().numpy().transpose(1, 2, 0)
            true_mask  = true_mask_tensor.cpu().numpy().squeeze()
            pred_mask  = preds_binary[0].cpu().numpy().squeeze()

            processed_input_display = np.clip(input_image_for_display[:, :, :3], 0, 1)

            fig, axes = plt.subplots(1, 5, figsize=(25, 6))

            axes[0].imshow(original_eo_display)
            axes[0].set_title('Original EO (Pre-event)')
            axes[0].axis('off')

            axes[1].imshow(original_sar_display, cmap='gray')
            axes[1].set_title('Original SAR (Post-event)')
            axes[1].axis('off')

            axes[2].imshow(processed_input_display)
            axes[2].set_title('Model Input (Processed EO)')
            axes[2].axis('off')

            axes[3].imshow(true_mask, cmap='gray')
            axes[3].set_title('Ground Truth Mask')
            axes[3].axis('off')

            axes[4].imshow(pred_mask, cmap='gray')
            axes[4].set_title(f"Predicted Mask (Thresh={CONFIG['threshold']})")
            axes[4].axis('off')

            plt.tight_layout()
            plt.show()

            samples_processed += 1


def evaluate_all_runs(runs, dataset_root, test_loader, test_dataset,
                      norm_stats):
    """
    Evaluate multiple checkpoint runs on the test set — exact logic from
    the training notebook's multi-run evaluation cell.

    Args:
        runs:          List of checkpoint folder names under dataset_root.
        dataset_root:  Root where checkpoint folders live.
        test_loader:   DataLoader for test split.
        test_dataset:  ChangeDetectionDataset for test split.
        norm_stats:    Normalisation stats dict.
    """
    from torch.utils.data import DataLoader
    from .dataset import ChangeDetectionDataset, _loader_kw

    for i in runs:
        ckpt = torch.load(
            os.path.join(dataset_root, i, 'best.pth'),
            map_location=device)

        original_encoder    = CONFIG['encoder']
        original_image_size = CONFIG['image_size']
        original_batch_size = CONFIG['batch_size']

        if 'config' in ckpt:
            CONFIG.update(ckpt['config'])
        else:
            if 'efficientnet' in i:
                CONFIG['encoder']     = 'efficientnet-b0'
                CONFIG['image_size']  = 512
            elif 'resnet50' in i:
                CONFIG['encoder']     = 'resnet50'
                CONFIG['image_size']  = 1024

        model     = get_model()
        raw_model = getattr(model, '_orig_mod', model)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        current_size = test_dataset.transform.transforms[0].height
        if CONFIG['image_size'] != current_size:
            print(f"Recreating test_loader for image_size {CONFIG['image_size']}")
            test_dataset = ChangeDetectionDataset(
                root_dir=dataset_root, split='test',
                norm_stats=norm_stats,
                image_size=CONFIG['image_size'], is_train=False
            )
            _kw = dict(
                batch_size=CONFIG['batch_size'],
                num_workers=CONFIG['num_workers'],
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
            test_loader = DataLoader(test_dataset, shuffle=False, **_kw)
            print(f"Test batches : {len(test_loader)}")

        test_loss, test_metrics, test_cm = test_model(model, test_loader)

        print(f"\nEvaluating: {i}")
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test F1:        {test_metrics['f1']:.4f}")
        print(f"Test IoU:       {test_metrics['iou']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall:    {test_metrics['recall']:.4f}")

        run_results_dir = os.path.join(dataset_root, i, 'test_results')
        os.makedirs(run_results_dir, exist_ok=True)

        test_results_path = os.path.join(run_results_dir, 'test_metrics.json')
        with open(test_results_path, 'w') as f:
            json.dump({
                'test_loss': test_loss,
                'metrics':   {k: v for k, v in test_metrics.items()},
                'confusion_matrix': test_cm.tolist()
            }, f, indent=4)
        print(f"Test results saved to: {test_results_path}")

        plot_test_confusion_matrix(test_cm, i, run_results_dir)

        CONFIG['encoder']     = original_encoder
        CONFIG['image_size']  = original_image_size
        CONFIG['batch_size']  = original_batch_size
