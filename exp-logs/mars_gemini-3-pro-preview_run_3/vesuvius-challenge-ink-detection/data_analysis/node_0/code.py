import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from scipy.stats import skew, kurtosis, pearsonr

# --- Configuration ---
INPUT_DIR = Path("./input")
METADATA_PATH = Path("./metadata/train.csv")
SEED = 42
SAMPLE_SIZE_PER_FRAGMENT = 50000  # Number of pixels to sample per fragment for stats
SLICES_TO_SAMPLE = [
    20,
    25,
    30,
    35,
    40,
    45,
]  # Middle slices usually contain the ink info


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_image(path, grayscale=True):
    path = str(path)
    if not os.path.exists(path):
        return None
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flags)


def run_eda():
    seed_everything(SEED)

    print("EDA Report")
    print("==========")

    # 1. Data Integrity
    # -----------------
    if not METADATA_PATH.exists():
        print("Error: Metadata file not found.")
        return

    df_train = pd.read_csv(METADATA_PATH)
    if df_train.empty:
        print("Error: Training metadata is empty.")
        return

    # 2. Target Variable Analysis
    # ---------------------------
    print("\nTARGET VARIABLE ANALYSIS")

    total_ink_pixels = 0
    total_valid_pixels = 0

    # We will also collect data for Feature Relationships here to avoid re-reading files
    # Lists to store sampled values
    sampled_pixel_intensities = []
    sampled_labels = []

    # Meta-feature lists
    fragment_areas = []
    ink_densities = []

    # Dimension lists
    widths = []
    heights = []
    aspect_ratios = []

    for _, row in df_train.iterrows():
        # Paths
        mask_rel = row["mask_path"]
        label_rel = row["inklabels_path"]
        vol_rel = row["surface_volume_path"]

        full_mask_path = INPUT_DIR / mask_rel
        full_label_path = INPUT_DIR / label_rel
        full_vol_dir = INPUT_DIR / vol_rel

        # Load Mask and Label
        mask = load_image(full_mask_path)
        label = load_image(full_label_path)

        if mask is None or label is None:
            continue

        # Dimensions
        h, w = mask.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        fragment_areas.append(w * h)

        # Flatten valid area
        valid_indices = np.where(mask > 0)
        valid_labels = label[valid_indices]

        # Binarize labels (just in case they aren't strictly 0/1, though usually they are)
        valid_labels_binary = (valid_labels > 0).astype(np.uint8)

        n_ink = np.sum(valid_labels_binary)
        n_valid = len(valid_labels_binary)

        total_ink_pixels += n_ink
        total_valid_pixels += n_valid

        ink_densities.append(n_ink / n_valid if n_valid > 0 else 0)

        # --- Sampling for Pixel Stats & Relationships ---
        # Select random indices from the valid pixels
        if n_valid > SAMPLE_SIZE_PER_FRAGMENT:
            sample_indices_idx = np.random.choice(
                n_valid, SAMPLE_SIZE_PER_FRAGMENT, replace=False
            )
            sampled_y = valid_indices[0][sample_indices_idx]
            sampled_x = valid_indices[1][sample_indices_idx]
            fragment_sampled_labels = valid_labels_binary[sample_indices_idx]
        else:
            sampled_y = valid_indices[0]
            sampled_x = valid_indices[1]
            fragment_sampled_labels = valid_labels_binary

        # Load specific Z-slices and extract pixel values
        # We average the intensity across the sampled slices for a robust feature representation
        fragment_pixel_values = np.zeros(len(sampled_y), dtype=np.float32)

        valid_slices_count = 0
        for z in SLICES_TO_SAMPLE:
            slice_filename = f"{z:02d}.tif"
            slice_path = full_vol_dir / slice_filename

            if slice_path.exists():
                img_slice = load_image(slice_path)
                if img_slice is not None:
                    # Extract values
                    vals = img_slice[sampled_y, sampled_x]
                    fragment_pixel_values += vals
                    valid_slices_count += 1

        if valid_slices_count > 0:
            fragment_pixel_values /= valid_slices_count

            sampled_pixel_intensities.extend(fragment_pixel_values)
            sampled_labels.extend(fragment_sampled_labels)

    # Calculate Target Stats
    ink_ratio = total_ink_pixels / total_valid_pixels if total_valid_pixels > 0 else 0
    no_ink_ratio = 1.0 - ink_ratio
    imbalance_ratio = no_ink_ratio / ink_ratio if ink_ratio > 0 else 0

    print(f"Target Variable: Ink Presence (Binary)")
    print(f"Global Ink Ratio: {ink_ratio:.4f}")
    print(f"Global No-Ink Ratio: {no_ink_ratio:.4f}")
    print(f"Class Imbalance Ratio (Neg/Pos): {imbalance_ratio:.4f}")

    # 3. Input Data Analysis (Image)
    # ------------------------------
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Dimensions
    w_series = pd.Series(widths)
    h_series = pd.Series(heights)
    ar_series = pd.Series(aspect_ratios)

    print(f"Fragment Count: {len(widths)}")
    print(
        f"Widths  - Mean: {w_series.mean():.4f}, Std: {w_series.std():.4f}, Min: {w_series.min():.4f}, Max: {w_series.max():.4f}"
    )
    print(
        f"Heights - Mean: {h_series.mean():.4f}, Std: {h_series.std():.4f}, Min: {h_series.min():.4f}, Max: {h_series.max():.4f}"
    )
    print(f"Aspect Ratios - Mean: {ar_series.mean():.4f}, Std: {ar_series.std():.4f}")

    # Channels
    # Based on task description, z-depth is 65.
    print(f"Channels (Z-Depth): 65 (Standardized across fragments)")

    # Pixel Stats (from sampled data)
    pixels_arr = np.array(sampled_pixel_intensities)
    print(f"Pixel Intensity (Sampled) - Mean: {np.mean(pixels_arr):.4f}")
    print(f"Pixel Intensity (Sampled) - Std : {np.std(pixels_arr):.4f}")

    # 4. Feature/Signal Relationships
    # -------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Structured Relationship: Pixel Intensity vs Label
    # We calculate Point-Biserial Correlation (Pearson between continuous and binary)
    labels_arr = np.array(sampled_labels)

    if len(pixels_arr) > 0 and len(labels_arr) > 0:
        # Check for non-constant input
        if np.std(pixels_arr) > 0 and np.std(labels_arr) > 0:
            corr, _ = pearsonr(pixels_arr, labels_arr)
            print(f"Pixel Intensity vs Ink Label Correlation: {corr:.4f}")

            # Simple importance check: Mean intensity of Ink vs No-Ink
            mean_ink = np.mean(pixels_arr[labels_arr == 1])
            mean_no_ink = np.mean(pixels_arr[labels_arr == 0])
            print(f"Mean Intensity (Ink Pixels): {mean_ink:.4f}")
            print(f"Mean Intensity (No-Ink Pixels): {mean_no_ink:.4f}")
            print(f"Signal Delta: {mean_ink - mean_no_ink:.4f}")
        else:
            print(
                "Pixel Intensity vs Ink Label Correlation: Undefined (Constant variance)"
            )
    else:
        print("Pixel Intensity vs Ink Label Correlation: N/A (No data sampled)")

    # Unstructured Relationship: Fragment Area vs Ink Density
    if len(fragment_areas) > 1:
        # Pearson correlation
        area_density_corr, _ = pearsonr(fragment_areas, ink_densities)
        print(f"Fragment Area vs Ink Density Correlation: {area_density_corr:.4f}")
    else:
        print("Fragment Area vs Ink Density Correlation: N/A (Insufficient fragments)")


if __name__ == "__main__":
    run_eda()
