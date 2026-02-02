import os
import pandas as pd
import numpy as np
import cv2
import random
import time

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Target is 'category_id'
    class_counts = df["category_id"].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    mean_count = class_counts.mean()
    std_count = class_counts.std()
    min_count = class_counts.min()
    max_count = class_counts.max()

    # Imbalance
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # Quantiles for distribution context
    quantiles = class_counts.quantile([0.25, 0.5, 0.75]).to_dict()

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")
    print(f"Class Distribution Stats:")
    print(f"  Mean Samples per Class: {mean_count:.4f}")
    print(f"  Std Dev Samples per Class: {std_count:.4f}")
    print(f"  Min Samples: {min_count}")
    print(f"  Max Samples: {max_count}")
    print(f"  25th Percentile: {quantiles[0.25]:.4f}")
    print(f"  Median: {quantiles[0.50]:.4f}")
    print(f"  75th Percentile: {quantiles[0.75]:.4f}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Return class counts for relationship analysis later
    return class_counts


def analyze_images(df):
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Subsample for speed
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Analyzing a subset of {len(sample_df)} images for pixel/dimension stats...")

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # For pixel stats (Welford's algorithm or simple accumulation)
    # Using simple accumulation for mean, and sum of squares for std
    # Accumulators for [R, G, B]
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    # Track metadata for relationship analysis
    meta_stats = []

    start_time = time.time()

    for idx, row in sample_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Check existence (though metadata generation verified this, good for robustness)
        if not os.path.exists(file_path):
            continue

        # Read image
        # cv2 reads in BGR
        img = cv2.imread(file_path)

        if img is None:
            continue

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels_list.append(c)

        # Pixel stats accumulation
        # Normalize to 0-1 for calculation to avoid overflow, then scale back or keep 0-1
        # Standard practice is often reporting 0-255 or 0-1. Let's do 0-255 stats.
        flat_img = img.reshape(-1, 3)
        n_pixels = flat_img.shape[0]

        pixel_sum += flat_img.sum(axis=0)
        pixel_sq_sum += (flat_img**2).sum(axis=0)
        pixel_count += n_pixels

        # Store for relationship analysis
        meta_stats.append(
            {
                "category_id": row["category_id"],
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
                "area": w * h,
            }
        )

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels = np.array(channels_list)

    print("Dimensions:")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    # Channels
    unique_channels, counts_channels = np.unique(channels, return_counts=True)
    print("Channels Distribution:")
    for c, count in zip(unique_channels, counts_channels):
        print(f"  {c} Channels: {count} images ({count/len(channels)*100:.2f}%)")

    # Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))

        # Normalize to 0-1 for the report as is common in ML (PyTorch)
        rgb_mean_norm = rgb_mean / 255.0
        rgb_std_norm = rgb_std / 255.0

        print("Pixel Statistics (RGB, 0-1 scale):")
        print(
            f"  Mean: [{rgb_mean_norm[0]:.4f}, {rgb_mean_norm[1]:.4f}, {rgb_mean_norm[2]:.4f}]"
        )
        print(
            f"  Std : [{rgb_std_norm[0]:.4f}, {rgb_std_norm[1]:.4f}, {rgb_std_norm[2]:.4f}]"
        )
    else:
        print("Pixel Statistics: N/A (No pixels processed)")

    return pd.DataFrame(meta_stats)


def analyze_relationships(meta_stats_df, class_counts):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    if meta_stats_df.empty:
        print("No image data available for relationship analysis.")
        return

    # Map class frequency to the image stats
    # class_counts is a Series index=category_id, value=count
    meta_stats_df["class_frequency"] = meta_stats_df["category_id"].map(class_counts)

    # 1. Unstructured (Meta-Feature) Relationships
    # Correlation between Image Metadata and Target Frequency
    # "Do common classes have larger images?" etc.

    correlations = meta_stats_df[
        ["width", "height", "aspect_ratio", "area", "class_frequency"]
    ].corr()

    print("Correlation between Image Metadata and Class Frequency:")
    print(f"  Width vs Freq: {correlations.loc['width', 'class_frequency']:.4f}")
    print(f"  Height vs Freq: {correlations.loc['height', 'class_frequency']:.4f}")
    print(f"  Area vs Freq: {correlations.loc['area', 'class_frequency']:.4f}")
    print(
        f"  Aspect Ratio vs Freq: {correlations.loc['aspect_ratio', 'class_frequency']:.4f}"
    )

    # Check if there is a significant difference in image size between rare and common classes
    # Define Rare < median frequency, Common >= median frequency
    median_freq = meta_stats_df["class_frequency"].median()
    rare_imgs = meta_stats_df[meta_stats_df["class_frequency"] < median_freq]
    common_imgs = meta_stats_df[meta_stats_df["class_frequency"] >= median_freq]

    print(
        f"\nGroup Comparison (Rare vs Common Classes, split at median freq {median_freq}):"
    )
    print(f"  Rare Class Mean Area: {rare_imgs['area'].mean():.4f}")
    print(f"  Common Class Mean Area: {common_imgs['area'].mean():.4f}")


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Target Analysis
    class_counts = analyze_target(df)

    # 3. Image Analysis
    meta_stats_df = analyze_images(df)

    # 4. Relationships
    analyze_relationships(meta_stats_df, class_counts)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
