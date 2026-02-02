import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import warnings
from collections import Counter

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
LABEL_MAP_PATH = "./input/label_num_to_disease_map.json"
SEED = 42
SAMPLE_SIZE_PIXEL_STATS = (
    1000  # Number of images to sample for pixel mean/std calculation
)
NUM_WORKERS = 12  # Based on 12 vCPUs available

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_image_metadata(file_path):
    """
    Extracts width, height, file size, and mode from an image file.
    Uses PIL to open lazily to avoid full decode overhead for dimensions.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    try:
        file_size = os.path.getsize(full_path)
        with Image.open(full_path) as img:
            width, height = img.size
            mode = img.mode
            return width, height, file_size, mode
    except Exception:
        return None, None, None, None


def compute_pixel_stats(file_paths):
    """
    Computes mean and std of pixels for a list of file paths.
    """
    sums = np.zeros(3)
    sq_sums = np.zeros(3)
    count = 0

    for fp in file_paths:
        full_path = os.path.join(INPUT_DIR, fp)
        try:
            with Image.open(full_path) as img:
                img = img.convert("RGB")
                # Resize to speed up stat calculation (256x256 is sufficient for estimation)
                img = img.resize((256, 256))
                arr = np.array(img) / 255.0

                sums += arr.sum(axis=(0, 1))
                sq_sums += (arr**2).sum(axis=(0, 1))
                count += arr.shape[0] * arr.shape[1]
        except Exception:
            continue

    if count == 0:
        return np.zeros(3), np.zeros(3)

    global_mean = sums / count
    global_std = np.sqrt((sq_sums / count) - (global_mean**2))

    return global_mean, global_std


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
        # Ensure keys are integers for mapping
        label_map = {int(k): v for k, v in label_map.items()}

    print("==========================================")
    print("       EXPLORATORY DATA ANALYSIS          ")
    print("==========================================")

    # ---------------------------------------------------------
    # 1. Target Variable Analysis
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    class_counts = df["label"].value_counts().sort_index()
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {len(class_counts)}")
    print("\nClass Distribution:")

    for label, count in class_counts.items():
        disease_name = label_map.get(label, "Unknown")
        ratio = count / total_samples
        print(f"Class {label} ({disease_name}): {count} samples ({ratio:.4f})")

    # Check for imbalance
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()
    imbalance_ratio = max_class_count / min_class_count
    print(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
    if imbalance_ratio > 5:
        print("Observation: The dataset is significantly imbalanced.")
    else:
        print("Observation: The dataset is moderately balanced.")

    # ---------------------------------------------------------
    # 2. Input Data Analysis (Image)
    # ---------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Extract metadata for all images using threads
    print("Extracting image metadata (Dimensions, Size)...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(get_image_metadata, df["file_path"]))

    # Unzip results
    widths, heights, file_sizes, modes = zip(*results)

    # Filter out Nones in case of read errors
    valid_indices = [i for i, w in enumerate(widths) if w is not None]
    widths = np.array([widths[i] for i in valid_indices])
    heights = np.array([heights[i] for i in valid_indices])
    file_sizes = np.array([file_sizes[i] for i in valid_indices])
    modes = [modes[i] for i in valid_indices]

    # Add to dataframe for relationship analysis later
    df_valid = df.iloc[valid_indices].copy()
    df_valid["width"] = widths
    df_valid["height"] = heights
    df_valid["file_size"] = file_sizes
    df_valid["aspect_ratio"] = widths / heights

    # Dimensions
    print("\nDimensions:")
    print(
        f"Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )

    # Aspect Ratios
    aspect_ratios = df_valid["aspect_ratio"]
    print("\nAspect Ratios:")
    print(f"Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")
    unique_ratios = np.unique(np.round(aspect_ratios, 4))
    if len(unique_ratios) < 10:
        print(f"Unique Aspect Ratios: {unique_ratios}")
    else:
        print(
            f"Most Common Aspect Ratios: {Counter(np.round(aspect_ratios, 2)).most_common(3)}"
        )

    # Channels
    print("\nChannels:")
    mode_counts = Counter(modes)
    for mode, count in mode_counts.items():
        print(f"Mode {mode}: {count} images ({count/len(modes):.4f})")

    # Pixel Stats (Sampled)
    print("\nPixel Statistics (RGB, Normalized 0-1):")
    if len(df) > SAMPLE_SIZE_PIXEL_STATS:
        sample_paths = (
            df["file_path"]
            .sample(n=SAMPLE_SIZE_PIXEL_STATS, random_state=SEED)
            .tolist()
        )
        print(
            f"Calculating stats on a random sample of {SAMPLE_SIZE_PIXEL_STATS} images..."
        )
    else:
        sample_paths = df["file_path"].tolist()
        print(f"Calculating stats on all {len(df)} images...")

    mean, std = compute_pixel_stats(sample_paths)
    print(f"Red   - Mean: {mean[0]:.4f}, Std: {std[0]:.4f}")
    print(f"Green - Mean: {mean[1]:.4f}, Std: {std[1]:.4f}")
    print(f"Blue  - Mean: {mean[2]:.4f}, Std: {std[2]:.4f}")

    # ---------------------------------------------------------
    # 3. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Group by label and check image properties
    print("Relationship between Disease Label and Image Metadata:")

    grouped = df_valid.groupby("label")[
        ["width", "height", "file_size", "aspect_ratio"]
    ].mean()

    # Format output
    print(
        f"{'Label':<6} | {'Disease':<35} | {'Avg Width':<10} | {'Avg Height':<10} | {'Avg Size(B)':<12} | {'Avg AR':<8}"
    )
    print("-" * 95)

    for label in grouped.index:
        disease = label_map.get(label, "Unknown")
        # Truncate disease name if too long
        if len(disease) > 34:
            disease = disease[:31] + "..."

        w = grouped.loc[label, "width"]
        h = grouped.loc[label, "height"]
        s = grouped.loc[label, "file_size"]
        ar = grouped.loc[label, "aspect_ratio"]

        print(
            f"{label:<6} | {disease:<35} | {w:<10.2f} | {h:<10.2f} | {s:<12.2f} | {ar:<8.4f}"
        )

    # Check for correlation between file size and label (using simple variance analysis logic)
    # We just check if any class deviates significantly from the global mean
    global_avg_size = df_valid["file_size"].mean()
    max_dev = 0
    max_dev_label = -1

    for label in grouped.index:
        diff = abs(grouped.loc[label, "file_size"] - global_avg_size)
        if diff > max_dev:
            max_dev = diff
            max_dev_label = label

    pct_dev = (max_dev / global_avg_size) * 100
    print(
        f"\nMax deviation in average file size: Class {max_dev_label} deviates by {pct_dev:.2f}% from global average."
    )
    if pct_dev > 10:
        print(
            "Observation: Some classes have noticeably different average file sizes/complexity."
        )
    else:
        print("Observation: File sizes are relatively consistent across classes.")


if __name__ == "__main__":
    main()
