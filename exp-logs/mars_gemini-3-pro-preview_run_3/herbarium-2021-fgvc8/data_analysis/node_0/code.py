import os
import cv2
import numpy as np
import pandas as pd
import random
import time
from collections import Counter
from scipy import stats

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_file_path(rel_path):
    return os.path.join(INPUT_DIR, rel_path)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "category_id"
    counts = df[target_col].value_counts()

    n_classes = len(counts)
    total_samples = len(df)
    min_count = counts.min()
    max_count = counts.max()
    mean_count = counts.mean()
    median_count = counts.median()

    # Imbalance Ratio
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {n_classes}")
    print(f"Distribution Stats:")
    print(f"  Min Samples per Class: {min_count}")
    print(f"  Max Samples per Class: {max_count}")
    print(f"  Mean Samples per Class: {mean_count:.4f}")
    print(f"  Median Samples per Class: {median_count:.4f}")
    print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 most frequent classes
    print(
        f"Top 5 Frequent Classes: {counts.head(5).index.tolist()} (Counts: {counts.head(5).values.tolist()})"
    )
    print("")
    return counts


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Sampling
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    else:
        sample_df = df.reset_index(drop=True)

    print(f"Analysis performed on a random sample of {len(sample_df)} images.")

    widths = []
    heights = []
    aspect_ratios = []
    channels_count = Counter()

    # For pixel stats (incremental calculation)
    # We will track sum and sum_sq per channel (assuming RGB max)
    # If images vary in channels, we handle them carefully.
    # We'll normalize everything to RGB for stats or report separately.
    # Given standard datasets, most are RGB.

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    file_sizes = []
    valid_samples = []  # Store indices of successfully loaded images

    start_time = time.time()

    for idx, row in sample_df.iterrows():
        full_path = get_file_path(row["file_path"])

        # Check file existence
        if not os.path.exists(full_path):
            continue

        # Get file size
        file_sizes.append(os.path.getsize(full_path))

        # Read Image
        # IMREAD_UNCHANGED to detect alpha or grayscale
        try:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
        except Exception:
            continue

        h, w = img.shape[:2]

        # Determine channels
        if len(img.shape) == 2:
            c = 1
            # Convert to RGB for consistent pixel stats
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            c = img.shape[2]
            if c == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif c == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                # Fallback for weird channel counts, just take first 3 or replicate
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if c > 3 else img

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels_count[c] += 1

        # Pixel Stats Accumulation
        # Normalize to 0-1 range for calculation to avoid huge numbers, then scale back or report in 0-255
        # Actually, standard is usually reporting in 0-255 or standardized. Let's do 0-255.

        flat_img = img_rgb.reshape(-1, 3).astype(np.float64)
        pixel_sum += flat_img.sum(axis=0)
        pixel_sq_sum += (flat_img**2).sum(axis=0)
        total_pixels += w * h

        valid_samples.append(idx)

    # Add extracted features back to sample_df for relationship analysis
    # We only keep rows that were valid
    sample_df = sample_df.iloc[valid_samples].copy()
    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["aspect_ratio"] = aspect_ratios
    sample_df["file_size_bytes"] = file_sizes
    sample_df["area"] = sample_df["width"] * sample_df["height"]

    # Statistics
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Dimensions:")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    print(f"Channels Distribution: {dict(channels_count)}")

    # Pixel Stats Calculation
    if total_pixels > 0:
        global_mean = pixel_sum / total_pixels
        global_var = (pixel_sq_sum / total_pixels) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print(f"Pixel Statistics (RGB, 0-255):")
        print(
            f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"  Std : R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
        )
    else:
        print("Pixel Statistics: N/A (No pixels processed)")

    print("")
    return sample_df


def analyze_relationships(sample_df, full_df):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # 1. Image Size vs Category (Top 5 Classes)
    # Do frequent classes have different image sizes?
    top_5_classes = full_df["category_id"].value_counts().head(5).index

    print("Relationship: Image Area vs Top 5 Categories")
    for cat_id in top_5_classes:
        subset = sample_df[sample_df["category_id"] == cat_id]
        if len(subset) > 0:
            mean_area = subset["area"].mean()
            std_area = subset["area"].std()
            print(
                f"  Class {cat_id}: Mean Area = {mean_area:.4f} (Std: {std_area:.4f}, n={len(subset)})"
            )
        else:
            print(f"  Class {cat_id}: No samples in subset.")

    # 2. File Size vs Image Area Correlation
    # Do larger images (pixels) correspond to larger file sizes (bytes)?
    # This checks compression/complexity indirectly.
    if len(sample_df) > 1:
        corr, _ = stats.pearsonr(sample_df["area"], sample_df["file_size_bytes"])
        print(f"\nMeta-Feature Correlation:")
        print(f"  Image Area (px) vs File Size (bytes): Pearson r = {corr:.4f}")

    # 3. Aspect Ratio vs Category
    # Check if aspect ratio varies significantly for the top class vs global
    global_ar_mean = sample_df["aspect_ratio"].mean()
    top_class = top_5_classes[0]
    top_class_subset = sample_df[sample_df["category_id"] == top_class]

    if len(top_class_subset) > 0:
        top_class_ar_mean = top_class_subset["aspect_ratio"].mean()
        print(f"\nAspect Ratio Comparison:")
        print(f"  Global Mean AR: {global_ar_mean:.4f}")
        print(f"  Class {top_class} Mean AR: {top_class_ar_mean:.4f}")

    print("")


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Run Analysis
    analyze_target(df)
    sample_df = analyze_images(df)
    analyze_relationships(sample_df, df)


if __name__ == "__main__":
    main()
