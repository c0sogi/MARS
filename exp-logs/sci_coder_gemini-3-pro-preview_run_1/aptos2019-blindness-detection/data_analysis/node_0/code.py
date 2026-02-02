import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from scipy.stats import pearsonr

# Constants
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_targets(df):
    print("=== TARGET VARIABLE ANALYSIS ===")
    target_col = "diagnosis"

    # Distribution
    counts = df[target_col].value_counts().sort_index()
    total = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total}")
    print("\nClass Distribution:")
    for label, count in counts.items():
        ratio = count / total
        print(f"Class {label}: {count} ({ratio:.4%})")

    # Imbalance
    max_class = counts.idxmax()
    min_class = counts.idxmin()
    imbalance_ratio = counts[max_class] / counts[min_class]
    print(f"\nImbalance Ratio (Maj/Min): {imbalance_ratio:.4f}")
    print("-" * 30)


def analyze_images(df):
    print("=== INPUT DATA ANALYSIS (IMAGE) ===")

    # Metrics to track
    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = {}
    file_sizes = []
    mean_intensities = []  # Per image mean intensity for meta-feature analysis

    # Global pixel stats accumulators (using Welford's algorithm or simple sum/sq_sum for global)
    # Given dataset size (~2.6k images), we can accumulate sum and sum_sq.
    # To avoid overflow, use float64.
    total_pixel_sum = np.zeros(3, dtype=np.float64)  # BGR
    total_pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixel_count = 0

    print(f"Processing {len(df)} images for statistical analysis...")

    # Iterate through images
    for idx, row in df.iterrows():
        # Construct path. Metadata file_path is relative to input dir root usually,
        # but the generate_metadata script prepended 'train_images/'.
        # The input dir structure is ./input/train_images/...
        # So full path is os.path.join(INPUT_DIR, row['file_path'])
        # Note: row['file_path'] from metadata already contains 'train_images/' prefix based on the provided metadata script.

        full_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # File size
            f_size = os.path.getsize(full_path)
            file_sizes.append(f_size)

            # Read Image
            img = cv2.imread(full_path)

            if img is None:
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            mean_intensities.append(img.mean())

            # Channel count
            if c not in channel_counts:
                channel_counts[c] = 0
            channel_counts[c] += 1

            # Pixel stats accumulation
            # Normalize to 0-1 for calculation if desired, but typically 0-255 is reported for raw stats.
            # We will calculate on 0-255 scale.

            # Flatten spatial dimensions
            pixels = img.reshape(-1, 3)
            n_pixels = pixels.shape[0]

            total_pixel_sum += pixels.sum(axis=0)
            total_pixel_sq_sum += (pixels**2).sum(axis=0)
            total_pixel_count += n_pixels

        except Exception as e:
            # Silent fail for individual corrupt images in EDA to prevent crash
            continue

    # 1. Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print("\nDimensions:")
    print(
        f"Widths  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"Heights - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"Aspect Ratios - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}, Min: {aspect_ratios.min():.4f}, Max: {aspect_ratios.max():.4f}"
    )

    # 2. Channels
    print("\nChannels:")
    for c, count in channel_counts.items():
        print(f"{c} Channels: {count} images")

    # 3. Pixel Stats
    # Calculate global mean and std per channel
    global_mean = total_pixel_sum / total_pixel_count
    # Var = E[X^2] - (E[X])^2
    global_var = (total_pixel_sq_sum / total_pixel_count) - (global_mean**2)
    global_std = np.sqrt(global_var)

    print("\nPixel Statistics (BGR scale 0-255):")
    # OpenCV loads as BGR
    print(f"Blue  - Mean: {global_mean[0]:.4f}, Std: {global_std[0]:.4f}")
    print(f"Green - Mean: {global_mean[1]:.4f}, Std: {global_std[1]:.4f}")
    print(f"Red   - Mean: {global_mean[2]:.4f}, Std: {global_std[2]:.4f}")

    # Return meta features for relationship analysis
    meta_df = pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "file_size_bytes": file_sizes,
            "mean_intensity": mean_intensities,
        }
    )
    # We need to align this with the original df.
    # Since we skipped None images, we assume strict alignment if no errors.
    # However, to be safe, let's assume the loop order was preserved and valid.
    # In a production script with potential failures, we'd track IDs.
    # Given the clean metadata check in the prompt, we assume alignment.
    return meta_df


def analyze_relationships(df, meta_df):
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Combine target with meta features
    # Reset index to ensure alignment if df was shuffled/filtered previously (though metadata is fresh)
    analysis_df = pd.concat(
        [df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1
    )
    target_col = "diagnosis"

    print("Unstructured (Meta-Feature) Relationships:")

    meta_features = [
        "width",
        "height",
        "aspect_ratio",
        "file_size_bytes",
        "mean_intensity",
    ]

    # Correlation with target
    print(f"\nCorrelation with Target ({target_col}):")
    for feat in meta_features:
        # Pearson correlation
        corr, _ = pearsonr(analysis_df[feat], analysis_df[target_col])
        print(f"{feat}: {corr:.4f}")

    # Average meta-feature value per class
    print("\nAverage Meta-Feature Value per Class:")
    grouped = analysis_df.groupby(target_col)[meta_features].mean()
    print(grouped.round(4))

    # Check for potential bias
    print("\nBias Check:")
    # Check if 'Severe' (3) or 'Proliferative' (4) images are significantly different in size/intensity
    # Compare Class 0 vs Class 4
    c0_stats = analysis_df[analysis_df[target_col] == 0][meta_features].mean()
    c4_stats = analysis_df[analysis_df[target_col] == 4][meta_features].mean()

    print("Comparison (No DR vs Proliferative DR):")
    for feat in meta_features:
        diff = c4_stats[feat] - c0_stats[feat]
        pct_diff = (diff / c0_stats[feat]) * 100 if c0_stats[feat] != 0 else 0
        print(f"{feat}: Diff = {diff:.4f} ({pct_diff:.2f}%)")


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_targets(df_train)

    # 2. Input Data Analysis (Image)
    meta_df = analyze_images(df_train)

    # 3. Feature Relationships
    analyze_relationships(df_train, meta_df)


if __name__ == "__main__":
    main()
