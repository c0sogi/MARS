import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import skew, kurtosis

# Configuration
SEED = 42
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")

    # In segmentation, the target is the mask.
    # We analyze the 'coverage' (ratio of salt pixels) as the target variable distribution.
    coverage = df["coverage"]

    # 1. Distribution
    print(f"Target Coverage Mean: {coverage.mean():.4f}")
    print(f"Target Coverage Std: {coverage.std():.4f}")
    print(f"Target Coverage Min: {coverage.min():.4f}")
    print(f"Target Coverage Max: {coverage.max():.4f}")

    # 2. Imbalance/Skew
    # For segmentation, class balance is the ratio of foreground (salt) to background (sediment) pixels.
    global_salt_ratio = coverage.mean()
    print(f"Global Class Balance (Salt): {global_salt_ratio:.4f}")
    print(f"Global Class Balance (Sediment): {(1 - global_salt_ratio):.4f}")

    # Skewness and Kurtosis of the coverage distribution
    print(f"Target Coverage Skewness: {skew(coverage):.4f}")
    print(f"Target Coverage Kurtosis: {kurtosis(coverage):.4f}")
    print("-" * 30)


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")

    image_paths = df["image_path"].tolist()

    widths = []
    heights = []
    channels = []
    aspect_ratios = []

    # Accumulators for global pixel stats
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    # Store mean intensity per image for relationship analysis
    img_mean_intensities = []

    # Iterate through all training images
    # Note: Dataset is small (~2400 images), so we can process all without sampling.
    for p in image_paths:
        full_path = os.path.join(INPUT_DIR, p)

        # Load image unchanged to detect channels correctly
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # Dimensions
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        channels.append(c)
        aspect_ratios.append(w / h)

        # Pixel Stats
        # Flatten to 1D array
        flat = img.flatten()

        # Update accumulators
        pixel_sum += np.sum(flat)
        pixel_sq_sum += np.sum(flat**2)
        total_pixels += len(flat)

        # Store local mean for meta-feature analysis
        img_mean_intensities.append(np.mean(flat))

    # 1. Dimensions
    print(f"Image Width Mean: {np.mean(widths):.4f}")
    print(f"Image Width Std: {np.std(widths):.4f}")
    print(f"Image Height Mean: {np.mean(heights):.4f}")
    print(f"Image Height Std: {np.std(heights):.4f}")
    print(f"Aspect Ratio Mean: {np.mean(aspect_ratios):.4f}")

    # 2. Channels
    unique_channels, counts = np.unique(channels, return_counts=True)
    channel_dist = dict(zip(unique_channels, counts))
    print(f"Channel Distribution (Channels: Count): {channel_dist}")

    # 3. Pixel Stats
    global_mean = pixel_sum / total_pixels
    global_var = (pixel_sq_sum / total_pixels) - (global_mean**2)
    global_std = np.sqrt(global_var)

    print(f"Global Pixel Mean: {global_mean:.4f}")
    print(f"Global Pixel Std: {global_std:.4f}")
    print("-" * 30)

    return img_mean_intensities


def analyze_relationships(df, img_intensities):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Add image intensities to dataframe for correlation
    # Ensure lengths match (they should if no images failed to load)
    if len(img_intensities) == len(df):
        df["mean_intensity"] = img_intensities
    else:
        # Fallback if some images were missing (unlikely based on metadata check)
        print("Warning: Image count mismatch for correlation analysis.")
        return

    # 1. Unstructured (Meta-Feature) Relationships

    # Relationship: Depth (z) vs Salt Coverage
    # Does salt appear more frequently at certain depths?
    corr_depth_cov = df["z"].corr(df["coverage"])
    print(f"Correlation (Depth vs Salt Coverage): {corr_depth_cov:.4f}")

    # Relationship: Image Intensity vs Salt Coverage
    # Is salt brighter or darker than sediment on average?
    corr_int_cov = df["mean_intensity"].corr(df["coverage"])
    print(f"Correlation (Image Intensity vs Salt Coverage): {corr_int_cov:.4f}")

    # Relationship: Depth vs Image Intensity
    # Does the seismic imaging change brightness with depth?
    corr_depth_int = df["z"].corr(df["mean_intensity"])
    print(f"Correlation (Depth vs Image Intensity): {corr_depth_int:.4f}")

    # 2. Structured Relationships (Tabular check on Metadata)
    # Check for missing values in the metadata columns we used
    print("\nMETADATA INTEGRITY")
    missing_z = df["z"].isna().sum()
    missing_cov = df["coverage"].isna().sum()
    print(f"Missing Values in Depth (z): {missing_z}")
    print(f"Missing Values in Coverage: {missing_cov}")


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Execute Analysis Modules
    analyze_target(df)
    img_intensities = analyze_images(df)
    analyze_relationships(df, img_intensities)


if __name__ == "__main__":
    main()
