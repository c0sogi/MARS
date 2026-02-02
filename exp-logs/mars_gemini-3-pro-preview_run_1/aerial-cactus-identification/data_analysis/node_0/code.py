import os
import pandas as pd
import numpy as np
import cv2
import sys
from scipy.stats import pointbiserialr

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train_metadata.csv"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)


def analyze_target_variable(df):
    """
    Analyzes the distribution and balance of the target variable.
    """
    print("SECTION: TARGET VARIABLE ANALYSIS")

    target_col = "has_cactus"
    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total}")

    # Distribution
    for label, count in counts.items():
        ratio = count / total
        print(f"Class {label}: {count} samples ({ratio:.4f})")

    # Imbalance
    if len(counts) == 2:
        # Assuming 0 and 1
        maj_class = counts.idxmax()
        min_class = counts.idxmin()
        ratio = counts[maj_class] / counts[min_class]
        print(f"Class Balance Ratio (Majority/Minority): {ratio:.4f}")
    else:
        print("Class Balance: Multiclass or Single class detected.")
    print("-" * 30)


def analyze_image_data(df):
    """
    Iterates through images to calculate dimensions, channel stats, and global pixel stats.
    Returns a dataframe enriched with meta-features for relationship analysis.
    """
    print("SECTION: INPUT DATA ANALYSIS (IMAGE)")

    # Accumulators for global stats
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Lists for meta-features
    widths = []
    heights = []
    aspect_ratios = []
    img_means = []  # Global mean intensity of the image
    img_stds = []  # Global contrast of the image
    file_sizes = []

    # Channel counters
    channel_counts = {}

    # Iterate through images
    # We use the file_path from metadata which is relative to input dir
    # e.g., train/id.jpg

    # Pre-calculate full paths
    image_paths = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x)).tolist()

    print(f"Processing {len(image_paths)} images...")

    valid_indices = []

    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):
            continue

        # Read image
        try:
            img = cv2.imread(path)
            if img is None:
                continue
        except Exception:
            continue

        valid_indices.append(idx)

        # Dimensions
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)

        # Channels distribution
        channel_counts[c] = channel_counts.get(c, 0) + 1

        # File size
        file_sizes.append(os.path.getsize(path))

        # Pixel Stats for Global Calculation
        # Convert to RGB for consistency if needed, but CV2 is BGR.
        # We will report stats in BGR order or just generic channel stats.
        # Let's assume RGB/BGR consistency across dataset.

        if c == 3:
            # Normalize to 0-1 for accumulation to avoid overflow if needed,
            # but float64 accumulation of 0-255 is fine for this dataset size.
            img_data = img.astype(np.float64)

            # Accumulate sum and squared sum per channel (B, G, R)
            # Reshape to (-1, 3)
            pixels = img_data.reshape(-1, 3)
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels**2).sum(axis=0)
            pixel_count += pixels.shape[0]

            # Meta-features per image (average intensity and contrast)
            img_means.append(img_data.mean())
            img_stds.append(img_data.std())
        else:
            # Handle grayscale if present
            img_data = img.astype(np.float64)
            pixels = img_data.reshape(-1, 1)
            # Add to all channels or handle separately?
            # Given the dataset description (photos), likely all RGB.
            # If grayscale, we treat as single channel.
            # For simplicity in this report, we skip global accumulation for mixed types
            # or assume RGB. Based on 'aerial photos', RGB is expected.
            img_means.append(img_data.mean())
            img_stds.append(img_data.std())

    # 1. Dimensions Analysis
    w_series = pd.Series(widths)
    h_series = pd.Series(heights)
    ar_series = pd.Series(aspect_ratios)

    print("Dimensions:")
    print(
        f"Width  - Mean: {w_series.mean():.4f}, Std: {w_series.std():.4f}, Min: {w_series.min()}, Max: {w_series.max()}"
    )
    print(
        f"Height - Mean: {h_series.mean():.4f}, Std: {h_series.std():.4f}, Min: {h_series.min()}, Max: {h_series.max()}"
    )
    print(f"Aspect Ratio - Mean: {ar_series.mean():.4f}, Std: {ar_series.std():.4f}")

    # 2. Channels
    print("\nChannels Distribution:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images ({count/len(valid_indices):.4f})")

    # 3. Global Pixel Stats
    if pixel_count > 0:
        # Calculate mean and std per channel
        # channel_sum is [Sum_B, Sum_G, Sum_R]
        global_mean = channel_sum / pixel_count
        global_std = np.sqrt((channel_sq_sum / pixel_count) - (global_mean**2))

        # Reorder BGR to RGB for reporting
        rgb_mean = global_mean[::-1]
        rgb_std = global_std[::-1]

        print("\nGlobal Pixel Statistics (RGB) [0-255 scale]:")
        print(f"  Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
        print(f"  Std : R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")
    else:
        print("\nGlobal Pixel Statistics: Could not compute (no RGB data processed).")

    # Return meta-features aligned with the valid indices
    # Create a DataFrame for the processed images
    meta_df = df.iloc[valid_indices].copy()
    meta_df["img_width"] = widths
    meta_df["img_height"] = heights
    meta_df["img_mean_intensity"] = img_means
    meta_df["img_contrast"] = img_stds
    meta_df["file_size_bytes"] = file_sizes

    print("-" * 30)
    return meta_df


def analyze_relationships(df):
    """
    Analyzes relationships between extracted meta-features and the target.
    """
    print("SECTION: FEATURE/SIGNAL RELATIONSHIPS")

    target_col = "has_cactus"

    # Meta-features to analyze
    features = ["img_mean_intensity", "img_contrast", "file_size_bytes"]

    print(f"Analyzing correlation with target '{target_col}' (Point-Biserial):")

    for feat in features:
        if feat in df.columns:
            # Calculate Point-Biserial Correlation
            # (Correlation between a binary variable and a continuous variable)
            corr, p_val = pointbiserialr(df[target_col], df[feat])
            print(f"  {feat}: Correlation = {corr:.4f}, P-value = {p_val:.4f}")

            # Grouped means for interpretation
            means = df.groupby(target_col)[feat].mean()
            print(f"    Mean ({target_col}=0): {means.get(0, 0):.4f}")
            print(f"    Mean ({target_col}=1): {means.get(1, 0):.4f}")

    print("-" * 30)


def main():
    set_seed(SEED)

    # Check for metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    # Load Data
    try:
        df = pd.read_csv(METADATA_FILE)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 1. Target Analysis
    analyze_target_variable(df)

    # 2. Image Analysis
    # This returns a dataframe with added meta-features for the next step
    df_enriched = analyze_image_data(df)

    # 3. Relationship Analysis
    if not df_enriched.empty:
        analyze_relationships(df_enriched)
    else:
        print("No valid image data found to analyze relationships.")


if __name__ == "__main__":
    main()
