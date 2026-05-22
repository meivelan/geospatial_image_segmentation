import zipfile
import os
import rasterio
from PIL import Image
import matplotlib as plt

# 1
def extract(zfile_path, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    try:
        with zipfile.ZipFile(zfile_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Successfully extracted '{zfile_path}' to '{destination_folder}'")

    except FileNotFoundError:
        print(f"Error: The file '{zfile_path}' was not found. Please check the path.")
    except zipfile.BadZipFile:
        print(f"Error: '{zfile_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 2
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
                        # Attempt to read a small part of the data to ensure it's readable
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

# 3
def check_dim_img():
    sizes = set()
    for f in os.listdir(train + "pre-event/")[:50]:
        with rasterio.open(train + "pre-event/" + f) as src:
            sizes.add((src.width, src.height))

    print("Unique sizes:", sizes)

# 4
def print_unique_tif_values(tif_image_path):
    """
    Opens a TIFF image and prints its unique pixel values.
    Args:
        tif_image_path (str): The path to the TIFF image.
    """
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

# 5
def show_tif_image_on_ax(ax, tif_image_path):
    try:
        img = Image.open(tif_image_path)
        ax.imshow(img)
        ax.set_title(
            f"{tif_image_path.split('/')[-1].split('_')[0].capitalize()}-event"
        )
        ax.axis("off")
    except FileNotFoundError:
        ax.set_title(f"Error: File not found: {tif_image_path.split('/')[-1]}")
        ax.axis("off")
    except Exception as e:
        print(8)
        ax.set_title(f"Error: {e}")
        ax.axis("off")

# 6
def calculate_change_percentage(mask_path):
    """
    Calculates the percentage of pixels with value 1 (changed) in a binary TIFF mask.
    Args:
        mask_path (str): The path to the binary TIFF mask image.
    Returns:
        float: The percentage of changed pixels, or None if an error occurs.
    """
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

# 7
def show_five_samples():
    files = os.listdir(train + "/pre-event/")
    samples = random.sample(files, 5)

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for i, fname in enumerate(samples):
        with rasterio.open(f"{train}/pre-event/{fname}") as src:
            eo = src.read([1, 2, 3]).transpose(1, 2, 0)
            eo = (eo - eo.min()) / (eo.max() - eo.min())

        with rasterio.open(f"{train}/post-event/{fname}") as src:
            sar = src.read(1)
            sar = (sar - sar.min()) / (sar.max() - sar.min())

        with rasterio.open(f"{train}/target/{fname}") as src:
            mask = src.read(1)

        axes[i, 0].imshow(eo)
        axes[i, 0].set_title("EO (pre-event)")
        axes[i, 1].imshow(sar, cmap="gray")
        axes[i, 1].set_title("SAR (post-event)")
        axes[i, 2].imshow(mask, cmap="Reds")
        axes[i, 2].set_title(f"Mask (change: {mask.mean()*100:.2f}%)")

    plt.tight_layout()
    plt.savefig("sample_visualization.png", dpi=100)
    plt.show()