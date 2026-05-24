import zipfile
import os
import rasterio
import numpy as np
import json
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from .config import CONFIG

dataset_root = CONFIG["dataset_root"]
train = dataset_root + "/train/"
test = dataset_root + "/test/"
val = dataset_root + "/val/"


def extract(zfile_path, destination_folder=None):
    if destination_folder is None:
        destination_folder = dataset_root
    os.makedirs(destination_folder, exist_ok=True)
    try:
        with zipfile.ZipFile(zfile_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Successfully extracted '{zfile_path}' to '{destination_folder}'")
    except FileNotFoundError:
        print(f"Error: The file '{zfile_path}' was not found.")
    except zipfile.BadZipFile:
        print(f"Error: '{zfile_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def re_labeling(target_input_directory):
    label_map = {0: 0, 1: 0, 2: 1, 3: 1}

    parent_directory = os.path.dirname(target_input_directory)
    re_labelled_output_directory = os.path.join(parent_directory, "re_labelled-target")

    os.makedirs(re_labelled_output_directory, exist_ok=True)
    print(f"Re-labeled images will be saved to: {re_labelled_output_directory}")

    for filename in os.listdir(target_input_directory):
        if filename.endswith((".tif", ".tiff")):
            input_filepath = os.path.join(target_input_directory, filename)
            output_filepath = os.path.join(re_labelled_output_directory, filename)
            try:
                with rasterio.open(input_filepath) as src:
                    img_array = src.read(1)
                    profile = src.profile.copy()

                    re_labeled_array = np.vectorize(label_map.get)(img_array)

                    profile.update(
                        dtype=rasterio.uint8,
                        TILED=True,
                        blockxsize=1024,
                        blockysize=1024,
                    )

                    with rasterio.open(output_filepath, "w", **profile) as dst:
                        dst.write(re_labeled_array.astype(rasterio.uint8), 1)
                print(
                    f"Successfully re-labeled and saved: {filename} to {re_labelled_output_directory}"
                )

            except rasterio.errors.RasterioIOError as e:
                print(f"Error: Rasterio encountered an issue with {filename}: {e}")
            except FileNotFoundError:
                print(f"Error: File not found: {input_filepath}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")


def print_unique_tif_values(tif_image_path):
    try:
        with rasterio.open(tif_image_path) as src:
            img_array = src.read(1)
            unique_values = np.unique(img_array)
            print(f"Unique values in {tif_image_path.split('/')[-1]}: {unique_values}")
    except FileNotFoundError:
        print(f"Error: File not found: {tif_image_path.split('/')[-1]}")
    except Exception as e:
        print(
            f"An error occurred while processing {tif_image_path.split('/')[-1]}: {e}"
        )


def inspect_image_metadata(image_path, image_type):
    try:
        with rasterio.open(image_path) as src:
            print(f"\n--- {image_type} Image: {image_path.split('/')[-1]} ---")
            print(f"  Dimensions: {src.width}x{src.height}")
            print(f"  Number of bands: {src.count}")
            print(f"  Data type: {src.dtypes[0]}")
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
    except Exception as e:
        print(f"An error occurred while inspecting {image_path}: {e}")


def calculate_change_percentage(mask_path):
    try:
        with rasterio.open(mask_path) as src:
            mask_array = src.read(1)
            total_pixels = mask_array.size
            changed_pixels = np.sum(mask_array == 1)
            if total_pixels == 0:
                return 0.0
            return (changed_pixels / total_pixels) * 100
    except Exception as e:
        print(f"Error processing {mask_path.split('/')[-1]}: {e}")
        return None


def check_dim_img(train_dir=None):
    t = train_dir or train
    sizes = set()
    for f in os.listdir(t + "pre-event/")[:50]:
        with rasterio.open(t + "pre-event/" + f) as src:
            sizes.add((src.width, src.height))
    print("Unique sizes:", sizes)


def valid__files(train_dir=None, root=None):
    t = train_dir or train
    r = root or dataset_root
    valid_files = []
    for f in os.listdir(f"{t}/target/"):
        with rasterio.open(f"{t}/target/{f}") as src:
            mask = src.read(1)
            if mask.sum() > 0:
                valid_files.append(f)

    print(f"Total train files: {len(os.listdir(f'{t}/target/'))}")
    print(f"Valid (non-empty) files: {len(valid_files)}")

    with open(r + "valid_train_files.txt", "w") as f:
        f.write("\n".join(valid_files))


def compute_and_save_diff(split_path):
    eo_dir = os.path.join(split_path, "pre-event/")
    sar_dir = os.path.join(split_path, "post-event/")
    diff_dir = os.path.join(split_path, "diff/")
    os.makedirs(diff_dir, exist_ok=True)

    print(f"Computing and saving diff images for {split_path.split('/')[-2]} split...")

    for fname in os.listdir(eo_dir):
        if not fname.endswith((".tif", ".tiff")):
            continue

        eo_file_path = os.path.join(eo_dir, fname)
        sar_file_path = os.path.join(sar_dir, fname)
        diff_file_path = os.path.join(diff_dir, fname)

        if os.path.exists(diff_file_path):
            print(f"Skipping {fname}: Difference file already exists.")
            continue

        try:
            with rasterio.open(eo_file_path) as src_eo:
                eo = src_eo.read().astype(np.float32)
                eo = eo / (255.0 if eo.max() > 1 else 1.0)
                profile = src_eo.profile.copy()

            with rasterio.open(sar_file_path) as src_sar:
                sar = src_sar.read(1).astype(np.float32)
                sar = sar / (sar.max() + 1e-8) if sar.max() > 0 else sar

            if eo.shape[0] >= 3:
                eo_gray = 0.299 * eo[0] + 0.587 * eo[1] + 0.114 * eo[2]
            elif eo.shape[0] == 1:
                eo_gray = eo[0]
            else:
                print(
                    f"Warning: Unexpected number of channels for EO image {fname}. Skipping."
                )
                continue

            diff = np.abs(eo_gray - sar)

            profile.update(
                driver="GTiff",
                count=1,
                dtype=rasterio.float32,
                compress="lzw",
                TILED=True,
                blockxsize=1024,
                blockysize=1024,
            )

            temp_diff_file_path = diff_file_path + ".tmp"
            with rasterio.open(temp_diff_file_path, "w", **profile) as dst:
                dst.write(diff[np.newaxis, :, :])
            os.rename(temp_diff_file_path, diff_file_path)
            print(f"Successfully computed and saved: {fname} to {diff_file_path}")

        except FileNotFoundError:
            print(f"Error: One of the files for {fname} not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {fname}: {e}")


def compute_pos_weight_for_all(train_dir=None, root=None):
    t = train_dir or train
    r = root or dataset_root
    re_labelled_target_dir = os.path.join(t, "re_labelled-target")
    sample_train_filenames = os.listdir(re_labelled_target_dir)

    total_positive_pixels = 0
    total_negative_pixels = 0

    print("Calculating pixel counts for pos_weight using all available files...")

    for i, filename in enumerate(sample_train_filenames):
        mask_path = os.path.join(re_labelled_target_dir, filename)
        try:
            with rasterio.open(mask_path) as src:
                mask_array = src.read(1).astype(np.uint8)
                total_positive_pixels += np.sum(mask_array == 1)
                total_negative_pixels += np.sum(mask_array == 0)
        except Exception as e:
            print(f"Error processing {filename} for pos_weight calculation: {e}")
        if i % 200 == 0:
            print(i)

    if total_positive_pixels == 0:
        print("Warning: No positive pixels found. pos_weight will be set to 1.")
        pos_weight_value_for_all = 1.0
    else:
        pos_weight_value_for_all = total_negative_pixels / total_positive_pixels

    print(f"Total Positive Pixels: {total_positive_pixels}")
    print(f"Total Negative Pixels: {total_negative_pixels}")
    print(f"Calculated pos_weight: {pos_weight_value_for_all:.2f}")
    with open(r + "pos_weight_for_all.txt", "w") as f:
        f.write(str(pos_weight_value_for_all))


def compute_pos_weight_for_valid(train_dir=None, root=None):
    t = train_dir or train
    r = root or dataset_root
    re_labelled_target_dir = os.path.join(t, "re_labelled-target")
    with open(r + "valid_train_files.txt", "r") as f:
        sample_train_filenames = f.read().split()

    total_positive_pixels = 0
    total_negative_pixels = 0

    print("Calculating pixel counts for pos_weight using valid files...")

    for i, filename in enumerate(sample_train_filenames):
        mask_path = os.path.join(re_labelled_target_dir, filename)
        try:
            with rasterio.open(mask_path) as src:
                mask_array = src.read(1).astype(np.uint8)
                total_positive_pixels += np.sum(mask_array == 1)
                total_negative_pixels += np.sum(mask_array == 0)
        except Exception as e:
            print(f"Error processing {filename} for pos_weight calculation: {e}")
        if i % 200 == 0:
            print(i)

    if total_positive_pixels == 0:
        print("Warning: No positive pixels found. pos_weight will be set to 1.")
        pos_weight_value_for_valid = 1.0
    else:
        pos_weight_value_for_valid = total_negative_pixels / total_positive_pixels

    print(f"Total Positive Pixels: {total_positive_pixels}")
    print(f"Total Negative Pixels: {total_negative_pixels}")
    print(f"Calculated pos_weight: {pos_weight_value_for_valid:.2f}")
    with open(r + "pos_weight_for_valid.txt", "w") as f:
        f.write(str(pos_weight_value_for_valid))


def check_corrupted_files(root_dir, split, include_diff=True):
    print(f"\n--- Checking for corrupted files in {split} split ---")

    pre_event_dir = os.path.join(root_dir, split, "pre-event")
    post_event_dir = os.path.join(root_dir, split, "post-event")
    target_dir = os.path.join(root_dir, split, "re_labelled-target")
    diff_dir = os.path.join(root_dir, split, "diff")

    directories_to_check = {
        "pre-event": pre_event_dir,
        "post-event": post_event_dir,
        "re_labelled-target": target_dir,
    }
    if include_diff:
        directories_to_check["diff"] = diff_dir

    corrupted_files_found = 0

    for dir_name, path in directories_to_check.items():
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' not found. Skipping {dir_name} check.")
            continue

        print(f"Checking {dir_name} directory...")
        for filename in os.listdir(path):
            if filename.endswith((".tif", ".tiff")):
                filepath = os.path.join(path, filename)
                try:
                    with rasterio.open(filepath) as src:
                        _ = src.read(1, window=((0, 1), (0, 1)))
                except rasterio.errors.RasterioIOError as e:
                    print(
                        f"  Corrupted file found in {dir_name}: {filename} - Error: {e}"
                    )
                    corrupted_files_found += 1
                except Exception as e:
                    print(
                        f"  Unexpected error with file in {dir_name}: {filename} - Error: {e}"
                    )
                    corrupted_files_found += 1

    if corrupted_files_found == 0:
        print(f"No corrupted files found in {split} split.")
    else:
        print(f"Total corrupted files found in {split} split: {corrupted_files_found}")


def filter_corrupted_valid_files(root=None):
    """Re-check valid_train_files.txt and remove any that can't be fully read."""
    r = root or dataset_root
    valid_files = open(r + "/valid_train_files.txt").read().splitlines()

    corrupted = []
    clean = []

    for fname in valid_files:
        try:
            with rasterio.open(f"{r}/train/pre-event/{fname}") as src:
                src.read(1)
            with rasterio.open(f"{r}/train/post-event/{fname}") as src:
                src.read(1)
            with rasterio.open(f"{r}/train/diff/{fname}") as src:
                src.read(1)
            with rasterio.open(f"{r}/train/re_labelled-target/{fname}") as src:
                src.read(1)
            clean.append(fname)
        except Exception as e:
            corrupted.append(fname)
            print(f"Corrupted: {fname} — {e}")

    print(f"\nTotal valid files: {len(valid_files)}")
    print(f"Corrupted: {len(corrupted)}")
    print(f"Clean: {len(clean)}")

    with open(r + "/valid_train_files.txt", "w") as f:
        f.write("\n".join(clean))
    print("Updated valid_train_files.txt")


def compute_norm_stats(root=None):
    """Compute mean/std for EO, SAR, diff over valid training files."""
    r = root or dataset_root
    valid_files = open(os.path.join(r, "valid_train_files.txt")).read().splitlines()

    eo_sum = np.zeros(3)
    eo_sq_sum = np.zeros(3)
    sar_sum = sar_sq_sum = diff_sum = diff_sq_sum = 0.0
    count = 0

    for i, fname in enumerate(valid_files):
        with rasterio.open(f"{r}/train/pre-event/{fname}") as src:
            eo = src.read().astype(np.float32) / 255.0
        with rasterio.open(f"{r}/train/post-event/{fname}") as src:
            sar = src.read(1).astype(np.float32) / 255.0
        with rasterio.open(f"{r}/train/diff/{fname}") as src:
            diff = src.read(1).astype(np.float32)

        eo_sum += eo.mean(axis=(1, 2))
        eo_sq_sum += (eo**2).mean(axis=(1, 2))
        sar_sum += sar.mean()
        sar_sq_sum += (sar**2).mean()
        diff_sum += diff.mean()
        diff_sq_sum += (diff**2).mean()
        count += 1
        if i % 200 == 0:
            print(i)

    n = count
    eo_mean = eo_sum / n
    eo_std = np.sqrt(eo_sq_sum / n - eo_mean**2)
    sar_mean = sar_sum / n
    sar_std = np.sqrt(sar_sq_sum / n - sar_mean**2)
    diff_mean = diff_sum / n
    diff_std = np.sqrt(diff_sq_sum / n - diff_mean**2)

    stats = {
        "eo_mean": eo_mean.tolist(),
        "eo_std": eo_std.tolist(),
        "sar_mean": float(sar_mean),
        "sar_std": float(sar_std),
        "diff_mean": float(diff_mean),
        "diff_std": float(diff_std),
    }

    with open(os.path.join(r, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nComputed from {count} valid training files")
    return stats
