import os
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis, pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    set_seed(42)
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    # Load training metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # -------------------------------------------------------------------------
    # 2. Target Variable Analysis
    # -------------------------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")

    # The target is the salt mask. We analyze the 'coverage' column which represents
    # the proportion of the image covered by salt.
    coverage = df_train["coverage"]

    # Distribution Statistics
    mean_cov = coverage.mean()
    std_cov = coverage.std()
    min_cov = coverage.min()
    max_cov = coverage.max()

    print(f"Target Variable: Salt Coverage (Proportion of pixels)")
    print(
        f"  Distribution: Mean={mean_cov:.4f}, Std={std_cov:.4f}, Min={min_cov:.4f}, Max={max_cov:.4f}"
    )

    # Skewness and Kurtosis (treating coverage as a regression target)
    skew_cov = skew(coverage)
    kurt_cov = kurtosis(coverage)
    print(f"  Normality Check: Skewness={skew_cov:.4f}, Kurtosis={kurt_cov:.4f}")

    # Class Balance (Binary: Empty Mask vs Non-Empty Mask)
    # An image is considered 'Empty' if coverage is 0
    empty_mask_count = (coverage == 0).sum()
    total_count = len(df_train)
    empty_ratio = empty_mask_count / total_count
    non_empty_ratio = 1.0 - empty_ratio

    print(f"  Class Balance (Binary - Has Salt vs No Salt):")
    print(f"    Empty Masks (No Salt): {empty_mask_count} ({empty_ratio:.4f})")
    print(
        f"    Non-Empty Masks: {total_count - empty_mask_count} ({non_empty_ratio:.4f})"
    )

    # -------------------------------------------------------------------------
    # 3. Input Data Analysis (Image Modality)
    # -------------------------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # We will load all training images to calculate global stats.
    # Given 2400 images of 101x101, this fits easily in memory.

    image_list = []
    widths = []
    heights = []
    channels = []

    # To calculate pixel stats efficiently
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    # Store per-image stats for relationship analysis later
    img_means = []
    img_stds = []

    # Iterate through training data
    # We use the relative path from metadata and prepend input dir
    for idx, row in df_train.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load image
        # Using IMREAD_UNCHANGED to detect if it's single channel or 3-channel
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        channels.append(c)

        # Flatten for stats
        img_flat = img.flatten()

        # Update global accumulators
        # Normalize to 0-1 range for standard reporting if strictly needed,
        # but usually raw pixel values (0-255) are reported. We report raw here.
        pixel_sum += np.sum(img_flat)
        pixel_sq_sum += np.sum(img_flat.astype(np.float64) ** 2)
        pixel_count += len(img_flat)

        # Per image stats
        img_means.append(np.mean(img_flat))
        img_stds.append(np.std(img_flat))

    # Convert lists to arrays for analysis
    widths = np.array(widths)
    heights = np.array(heights)
    channels = np.array(channels)

    # Dimensions
    print(f"  Dimensions:")
    print(
        f"    Width: Mean={widths.mean():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"    Height: Mean={heights.mean():.4f}, Min={heights.min()}, Max={heights.max()}"
    )

    # Aspect Ratios
    aspect_ratios = widths / heights
    print(
        f"    Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # Channels
    unique_channels, counts_channels = np.unique(channels, return_counts=True)
    print(f"  Channels Distribution:")
    for c, count in zip(unique_channels, counts_channels):
        print(f"    {c} Channel(s): {count} images ({count/len(widths):.4f})")

    # Pixel Stats (Global)
    global_mean = pixel_sum / pixel_count
    global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
    global_std = np.sqrt(global_var)

    print(f"  Pixel Statistics (Global, 0-255 scale):")
    print(f"    Mean: {global_mean:.4f}")
    print(f"    Std Dev: {global_std:.4f}")

    # -------------------------------------------------------------------------
    # 4. Feature/Signal Relationships
    # -------------------------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Add image stats to dataframe for correlation
    df_train["img_mean"] = img_means
    df_train["img_std"] = img_stds

    # 4.1 Unstructured (Meta-Feature) Relationships
    # Relationship between Metadata (Depth 'z') and Target ('coverage')
    corr_z_cov, _ = pearsonr(df_train["z"], df_train["coverage"])

    print(f"  Meta-Feature Relationships:")
    print(f"    Correlation (Depth 'z' vs Salt Coverage): {corr_z_cov:.4f}")

    # Relationship between Image Signal Properties and Target
    # Do brighter images or higher contrast images tend to have more salt?
    corr_mean_cov, _ = pearsonr(df_train["img_mean"], df_train["coverage"])
    corr_std_cov, _ = pearsonr(df_train["img_std"], df_train["coverage"])

    print(f"    Correlation (Image Pixel Mean vs Salt Coverage): {corr_mean_cov:.4f}")
    print(f"    Correlation (Image Pixel Std vs Salt Coverage): {corr_std_cov:.4f}")

    # Relationship between Depth and Image Intensity
    # Does depth affect the brightness of the seismic image?
    corr_z_mean, _ = pearsonr(df_train["z"], df_train["img_mean"])
    print(f"    Correlation (Depth 'z' vs Image Pixel Mean): {corr_z_mean:.4f}")


if __name__ == "__main__":
    main()
