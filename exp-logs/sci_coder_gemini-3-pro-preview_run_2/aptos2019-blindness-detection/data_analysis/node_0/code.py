import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import skew, kurtosis

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_stats(row):
    """
    Worker function to process a single image.
    Returns a dictionary of statistics or None if failed.
    """
    rel_path = row["file_path"]
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Dimensions
        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # Pixel Stats (accumulators for global calculation)
        # Convert to float for precision
        img_flat = img.reshape(-1, channels).astype(np.float64)

        # Per-channel sum and sum of squares
        pixel_sum = np.sum(img_flat, axis=0)
        pixel_sq_sum = np.sum(img_flat**2, axis=0)
        pixel_count = img_flat.shape[0]

        # Meta features for correlation analysis
        # We calculate local mean/std here for row-level features
        local_mean = np.mean(img_flat, axis=0).mean()  # Average across channels
        local_std = np.std(img_flat, axis=0).mean()

        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height > 0 else 0,
            "channels": channels,
            "pixel_sum": pixel_sum,  # Array of shape (C,)
            "pixel_sq_sum": pixel_sq_sum,  # Array of shape (C,)
            "pixel_count": pixel_count,
            "mean_intensity": local_mean,
            "std_intensity": local_std,
            "file_size_bytes": os.path.getsize(full_path),
        }
    except Exception:
        return None


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    print("=== DATA INTEGRITY ===")
    print(f"Analysis performed on Training Set only.")
    print(f"Number of samples: {len(df)}")

    # 2. Target Variable Analysis
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    target_col = "diagnosis"

    # Distribution
    class_counts = df[target_col].value_counts().sort_index()
    class_props = df[target_col].value_counts(normalize=True).sort_index()

    print("Class Distribution (Diagnosis):")
    for cls, count in class_counts.items():
        prop = class_props[cls]
        print(f"Class {cls}: {count} samples ({prop:.4%})")

    # Imbalance
    max_cls = class_counts.max()
    min_cls = class_counts.min()
    imbalance_ratio = max_cls / min_cls if min_cls > 0 else float("inf")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # 3. Input Data Analysis (Image Modality)
    print("\n=== INPUT DATA ANALYSIS (IMAGE) ===")

    # Use parallel processing to extract image stats
    # Convert dataframe to list of dicts for iteration
    rows = df.to_dict("records")

    stats_list = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(get_image_stats, rows))

    # Filter out Nones
    valid_results = [r for r in results if r is not None]
    failed_count = len(rows) - len(valid_results)

    if failed_count > 0:
        print(f"Warning: Failed to process {failed_count} images.")

    if not valid_results:
        print("No images processed successfully.")
        return

    # Create a DataFrame for meta-analysis
    df_stats = pd.DataFrame(valid_results)

    # Dimensions
    print("Dimensions:")
    print(
        f"Width:  Mean={df_stats['width'].mean():.4f}, Std={df_stats['width'].std():.4f}, "
        f"Min={df_stats['width'].min()}, Max={df_stats['width'].max()}"
    )
    print(
        f"Height: Mean={df_stats['height'].mean():.4f}, Std={df_stats['height'].std():.4f}, "
        f"Min={df_stats['height'].min()}, Max={df_stats['height'].max()}"
    )

    ar = df_stats["aspect_ratio"]
    print(
        f"Aspect Ratio: Mean={ar.mean():.4f}, Std={ar.std():.4f}, Min={ar.min():.4f}, Max={ar.max():.4f}"
    )

    # Channels
    channel_counts = df_stats["channels"].value_counts()
    print("Channel Counts:")
    for ch, count in channel_counts.items():
        print(f"{ch} Channels: {count} images")

    # Pixel Stats (Global)
    # We aggregate sum and sq_sum across all images
    # Note: Images might have different channel counts (though unlikely here, we handle it)
    # Assuming most are RGB (3 channels). If mixed, we analyze the dominant mode.

    # Filter for RGB images for global stats to be consistent
    rgb_stats = [r for r in valid_results if r["channels"] == 3]
    if rgb_stats:
        total_pixels = sum(r["pixel_count"] for r in rgb_stats)
        total_sum = np.sum([r["pixel_sum"] for r in rgb_stats], axis=0)
        total_sq_sum = np.sum([r["pixel_sq_sum"] for r in rgb_stats], axis=0)

        global_mean = total_sum / total_pixels
        # Var = E[X^2] - (E[X])^2
        global_var = (total_sq_sum / total_pixels) - (global_mean**2)
        global_std = np.sqrt(global_var)

        # OpenCV loads as BGR
        print("Global Pixel Statistics (BGR format):")
        print(
            f"Mean: B={global_mean[0]:.4f}, G={global_mean[1]:.4f}, R={global_mean[2]:.4f}"
        )
        print(
            f"Std:  B={global_std[0]:.4f},  G={global_std[1]:.4f},  R={global_std[2]:.4f}"
        )
        print(f"Pixel values are in range [0, 255]")
    else:
        print("No RGB images found for global pixel statistics.")

    # 4. Feature/Signal Relationships
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Merge stats back with diagnosis
    # We assume the order is preserved or we need to merge on index/id if we had it.
    # Since we iterated over 'rows' derived from 'df', and 'valid_results' corresponds to them
    # (minus failures), we need to be careful.
    # Let's reconstruct the DataFrame properly.

    # Add id_code to results for merging
    for i, res in enumerate(results):
        if res:
            res["id_code"] = rows[i]["id_code"]

    df_meta = pd.DataFrame([r for r in results if r is not None])
    df_merged = pd.merge(df, df_meta, on="id_code", how="inner")

    print("Unstructured (Meta-Feature) Relationships with Target:")

    # Correlation Analysis
    # Diagnosis is ordinal, so Spearman correlation is appropriate
    meta_cols = [
        "width",
        "height",
        "aspect_ratio",
        "file_size_bytes",
        "mean_intensity",
        "std_intensity",
    ]

    correlations = (
        df_merged[meta_cols + ["diagnosis"]]
        .corr(method="spearman")["diagnosis"]
        .drop("diagnosis")
    )

    print("Spearman Correlation with Diagnosis:")
    for feat, corr in correlations.items():
        print(f"{feat}: {corr:.4f}")

    # Grouped Analysis
    print("\nAverage Meta-Features by Class:")
    grouped = df_merged.groupby("diagnosis")[meta_cols].mean()
    print(grouped.round(4).to_string())

    # Check for specific insight: Do larger images correlate with specific classes?
    # (Already covered by correlation, but let's explicitly answer the prompt's example style)
    print("\nInsight Check:")
    width_corr = correlations["width"]
    if abs(width_corr) > 0.1:
        direction = "positive" if width_corr > 0 else "negative"
        print(
            f"There is a weak {direction} correlation ({width_corr:.4f}) between image width and diagnosis severity."
        )
    else:
        print(
            f"There is negligible correlation ({width_corr:.4f}) between image width and diagnosis severity."
        )


if __name__ == "__main__":
    main()
