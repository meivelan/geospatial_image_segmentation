"""
preprocessing.py
----------------
All dataset preparation steps that run *once* before training:

    1. re_label_targets   - remap class 0/1 → 0, class 2/3 → 1
    2. compute_diffs      - per-pixel |EO_gray - SAR| for every split
    3. find_valid_files   - list training tiles that have at least one
                            changed pixel (skip all-background tiles)
    4. compute_norm_stats - per-channel mean/std over the training set
    5. compute_pos_weight - negative / positive pixel ratio for BCE
    6. check_corrupted    - verify every file in each split is readable
    7. run_all            - convenience wrapper that runs 1-6 in order

Usage
-----
    from galaxeye.preprocessing import run_all
    run_all('/path/to/dataset_root')
"""

import os
import json
import warnings
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# ── Label remapping ────────────────────────────────────────────────────────

_LABEL_MAP = {0: 0, 1: 0, 2: 1, 3: 1}


def re_label_targets(dataset_root: str, splits=('train', 'val', 'test')) -> None:
    """
    Read raw target TIFFs (values 0-3) and write binary masks (0/1) to
    <split>/re_labelled-target/.

    Args:
        dataset_root: Root directory that contains train/, val/, test/.
        splits:       Which splits to process.
    """
    for split in splits:
        src_dir = os.path.join(dataset_root, split, 'target')
        dst_dir = os.path.join(dataset_root, split, 're_labelled-target')
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"[re_label] Skipping {split} - target dir not found: {src_dir}")
            continue

        files = [f for f in os.listdir(src_dir) if f.endswith(('.tif', '.tiff'))]
        print(f"[re_label] {split}: processing {len(files)} files → {dst_dir}")

        for filename in files:
            dst_path = os.path.join(dst_dir, filename)
            if os.path.exists(dst_path):
                continue  # already done

            src_path = os.path.join(src_dir, filename)
            try:
                with rasterio.open(src_path) as src:
                    arr = src.read(1)
                    profile = src.profile.copy()

                relabelled = np.vectorize(_LABEL_MAP.get)(arr).astype(np.uint8)
                profile.update(dtype=rasterio.uint8, TILED=True,
                                blockxsize=1024, blockysize=1024)

                tmp = dst_path + '.tmp'
                with rasterio.open(tmp, 'w', **profile) as dst:
                    dst.write(relabelled, 1)
                os.rename(tmp, dst_path)

            except Exception as e:
                print(f"  [re_label] ERROR {filename}: {e}")

        print(f"[re_label] {split}: done.")


# ── Difference images ──────────────────────────────────────────────────────

def compute_diffs(dataset_root: str, splits=('train', 'val', 'test')) -> None:
    """
    Compute and save |EO_gray - SAR_norm| per tile for each split.
    Output is a single-band float32 GeoTIFF in <split>/diff/.

    Args:
        dataset_root: Root directory containing the splits.
        splits:       Which splits to process.
    """
    for split in splits:
        eo_dir   = os.path.join(dataset_root, split, 'pre-event')
        sar_dir  = os.path.join(dataset_root, split, 'post-event')
        diff_dir = os.path.join(dataset_root, split, 'diff')
        os.makedirs(diff_dir, exist_ok=True)

        if not os.path.isdir(eo_dir):
            print(f"[diff] Skipping {split} - pre-event dir not found.")
            continue

        files = [f for f in os.listdir(eo_dir) if f.endswith(('.tif', '.tiff'))]
        print(f"[diff] {split}: computing {len(files)} difference images…")

        for fname in files:
            dst_path = os.path.join(diff_dir, fname)
            if os.path.exists(dst_path):
                continue

            try:
                with rasterio.open(os.path.join(eo_dir, fname)) as src:
                    eo      = src.read().astype(np.float32)
                    profile = src.profile.copy()
                eo = eo / (255.0 if eo.max() > 1 else 1.0)

                with rasterio.open(os.path.join(sar_dir, fname)) as src:
                    sar = src.read(1).astype(np.float32)
                sar_max = sar.max()
                sar = sar / (sar_max + 1e-8) if sar_max > 0 else sar

                if eo.shape[0] >= 3:
                    eo_gray = 0.299 * eo[0] + 0.587 * eo[1] + 0.114 * eo[2]
                else:
                    eo_gray = eo[0]

                diff = np.abs(eo_gray - sar)

                profile.update(driver='GTiff', count=1, dtype=rasterio.float32,
                                compress='lzw', TILED=True,
                                blockxsize=1024, blockysize=1024)

                tmp = dst_path + '.tmp'
                with rasterio.open(tmp, 'w', **profile) as dst:
                    dst.write(diff[np.newaxis])
                os.rename(tmp, dst_path)

            except Exception as e:
                print(f"  [diff] ERROR {fname}: {e}")

        print(f"[diff] {split}: done.")


# ── Valid file list ────────────────────────────────────────────────────────

def find_valid_files(dataset_root: str) -> list:
    """
    Return filenames (from train/re_labelled-target/) that have at least
    one changed pixel, and also pass a read-integrity check across all four
    modalities (pre, post, diff, re_labelled-target).

    Saves the result to <dataset_root>/valid_train_files.txt.

    Returns:
        List of valid filenames.
    """
    target_dir = os.path.join(dataset_root, 'train', 're_labelled-target')
    pre_dir    = os.path.join(dataset_root, 'train', 'pre-event')
    post_dir   = os.path.join(dataset_root, 'train', 'post-event')
    diff_dir   = os.path.join(dataset_root, 'train', 'diff')

    all_files = [f for f in os.listdir(target_dir)
                 if f.endswith(('.tif', '.tiff'))]

    print(f"[valid_files] Scanning {len(all_files)} training tiles…")
    valid = []

    for fname in all_files:
        try:
            with rasterio.open(os.path.join(target_dir, fname)) as src:
                mask = src.read(1)
            if mask.sum() == 0:
                continue  # skip background-only tiles

            # Integrity check
            for path in [
                os.path.join(pre_dir,    fname),
                os.path.join(post_dir,   fname),
                os.path.join(diff_dir,   fname),
                os.path.join(target_dir, fname),
            ]:
                with rasterio.open(path) as src:
                    src.read(1, window=((0, 1), (0, 1)))  # read 1 pixel

            valid.append(fname)

        except Exception as e:
            print(f"  [valid_files] Skipping {fname}: {e}")

    out_path = os.path.join(dataset_root, 'valid_train_files.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(valid))

    print(f"[valid_files] {len(valid)} / {len(all_files)} tiles are valid "
          f"→ saved to {out_path}")
    return valid


# ── Normalisation statistics ───────────────────────────────────────────────

def compute_norm_stats(dataset_root: str,
                       valid_files: list = None) -> dict:
    """
    Compute per-channel mean and std over the training set using Welford-style
    online accumulation (memory-efficient).

    Args:
        dataset_root: Dataset root directory.
        valid_files:  List of filenames to use. If None, reads
                      valid_train_files.txt from dataset_root.

    Returns:
        dict with keys eo_mean, eo_std, sar_mean, sar_std, diff_mean, diff_std.
        Also saves the result to <dataset_root>/norm_stats.json.
    """
    if valid_files is None:
        txt = os.path.join(dataset_root, 'valid_train_files.txt')
        with open(txt) as f:
            valid_files = f.read().splitlines()

    pre_dir  = os.path.join(dataset_root, 'train', 'pre-event')
    post_dir = os.path.join(dataset_root, 'train', 'post-event')
    diff_dir = os.path.join(dataset_root, 'train', 'diff')

    eo_sum   = np.zeros(3, dtype=np.float64)
    eo_sq    = np.zeros(3, dtype=np.float64)
    sar_sum  = sar_sq = diff_sum = diff_sq = 0.0
    n = 0

    print(f"[norm_stats] Computing stats over {len(valid_files)} tiles…")
    for i, fname in enumerate(valid_files):
        try:
            with rasterio.open(os.path.join(pre_dir, fname)) as src:
                eo = src.read().astype(np.float32) / 255.0   # (3, H, W)
            with rasterio.open(os.path.join(post_dir, fname)) as src:
                sar = src.read(1).astype(np.float32) / 255.0
            with rasterio.open(os.path.join(diff_dir, fname)) as src:
                diff = src.read(1).astype(np.float32)

            eo_sum  += eo.mean(axis=(1, 2))
            eo_sq   += (eo ** 2).mean(axis=(1, 2))
            sar_sum += sar.mean();   sar_sq  += (sar  ** 2).mean()
            diff_sum += diff.mean(); diff_sq += (diff ** 2).mean()
            n += 1

        except Exception as e:
            print(f"  [norm_stats] Skipping {fname}: {e}")

        if i % 200 == 0:
            print(f"  {i}/{len(valid_files)}", end='\r')

    eo_mean = eo_sum / n
    eo_std  = np.sqrt(np.maximum(eo_sq / n - eo_mean ** 2, 0))

    stats = {
        'eo_mean':   eo_mean.tolist(),
        'eo_std':    eo_std.tolist(),
        'sar_mean':  float(sar_sum / n),
        'sar_std':   float(np.sqrt(max(sar_sq / n - (sar_sum / n) ** 2, 0))),
        'diff_mean': float(diff_sum / n),
        'diff_std':  float(np.sqrt(max(diff_sq / n - (diff_sum / n) ** 2, 0))),
    }

    out_path = os.path.join(dataset_root, 'norm_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n[norm_stats] Done (n={n}). Saved → {out_path}")
    print(json.dumps(stats, indent=2))
    return stats


# ── Positive weight ────────────────────────────────────────────────────────

def compute_pos_weight(dataset_root: str, valid_files: list = None) -> float:
    """
    Compute neg_pixels / pos_pixels across re_labelled-target masks for the
    pos_weight argument of BCEWithLogitsLoss.

    Args:
        dataset_root: Dataset root directory.
        valid_files:  Filenames to include. If None, reads
                      valid_train_files.txt.

    Returns:
        pos_weight value (float). Also saved to pos_weight.txt.
    """
    if valid_files is None:
        txt = os.path.join(dataset_root, 'valid_train_files.txt')
        with open(txt) as f:
            valid_files = f.read().splitlines()

    target_dir = os.path.join(dataset_root, 'train', 're_labelled-target')
    pos = neg = 0

    print(f"[pos_weight] Scanning {len(valid_files)} masks…")
    for i, fname in enumerate(valid_files):
        try:
            with rasterio.open(os.path.join(target_dir, fname)) as src:
                mask = src.read(1).astype(np.uint8)
            pos += int(np.sum(mask == 1))
            neg += int(np.sum(mask == 0))
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
        if i % 200 == 0:
            print(f"  {i}/{len(valid_files)}", end='\r')

    w = neg / pos if pos > 0 else 1.0
    print(f"\n[pos_weight] pos={pos:,}  neg={neg:,}  weight={w:.2f}")

    with open(os.path.join(dataset_root, 'pos_weight.txt'), 'w') as f:
        f.write(str(w))

    return w


# ── Integrity check ────────────────────────────────────────────────────────

def check_corrupted(dataset_root: str,
                    splits=('train', 'val', 'test')) -> dict:
    """
    Try to open and read 1 pixel from every TIFF in pre-event, post-event,
    re_labelled-target, and diff for each split.

    Returns:
        Dict mapping split → list of corrupted filenames.
    """
    results = {}
    dirs_to_check = ['pre-event', 'post-event', 're_labelled-target', 'diff']

    for split in splits:
        bad = []
        print(f"\n[check_corrupted] {split}…")
        for sub in dirs_to_check:
            d = os.path.join(dataset_root, split, sub)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(('.tif', '.tiff')):
                    continue
                try:
                    with rasterio.open(os.path.join(d, fname)) as src:
                        src.read(1, window=((0, 1), (0, 1)))
                except Exception as e:
                    bad.append(f"{sub}/{fname}: {e}")

        results[split] = bad
        if bad:
            print(f"  ⚠ {len(bad)} corrupted file(s):")
            for b in bad:
                print(f"    {b}")
        else:
            print(f"  ✓ No corrupted files in {split}.")

    return results


# ── One-call convenience ───────────────────────────────────────────────────

def run_all(dataset_root: str, splits=('train', 'val', 'test')) -> dict:
    """
    Run the full preprocessing pipeline in order:
        1. re_label_targets
        2. compute_diffs
        3. find_valid_files
        4. compute_norm_stats
        5. compute_pos_weight
        6. check_corrupted

    Args:
        dataset_root: Root directory of the dataset.
        splits:       Which splits to preprocess.

    Returns:
        dict with keys: valid_files, norm_stats, pos_weight.
    """
    print("=" * 60)
    print("GalaxEye Preprocessing Pipeline")
    print("=" * 60)

    re_label_targets(dataset_root, splits)
    compute_diffs(dataset_root, splits)

    valid_files = find_valid_files(dataset_root)
    norm_stats  = compute_norm_stats(dataset_root, valid_files)
    pos_weight  = compute_pos_weight(dataset_root, valid_files)
    check_corrupted(dataset_root, splits)

    print("\n✓ Preprocessing complete.")
    return {
        'valid_files': valid_files,
        'norm_stats':  norm_stats,
        'pos_weight':  pos_weight,
    }
