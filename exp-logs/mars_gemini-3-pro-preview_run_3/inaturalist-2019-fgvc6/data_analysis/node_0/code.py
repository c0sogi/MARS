import os
import cv2
import numpy as np
import pandas as pd
import random
from collections import Counter

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train_metadata.csv"
SAMPLE_SIZE = 2500  # Number of images to sample for pixel/dimension stats
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_super_category(file_path):
    """
    Extracts the super category from the file path.
    Expected format: train_val2019/{SuperCategory}/{Category}/{ImageID}.jpg
    """
    parts = file_path.split(os.sep)
    # If path starts with train_val2019, the next part is likely the super category
    if len(parts) > 2:
        return parts[1]
    return "Unknown"


def analyze_targets(df):
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    # Distribution of category_id
    class_counts = df["category_id"].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")
    print(f"Class Distribution Stats:")
    print(f"  Min Samples per Class: {min_samples}")
    print(f"  Max Samples per Class: {max_samples}")
    print(f"  Mean Samples per Class: {mean_samples:.4f}")
    print(f"  Median Samples per Class: {median_samples:.4f}")

    # Imbalance Ratio (Max / Min)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float("inf")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 and Bottom 5 Classes
    print("Top 5 Frequent Classes (ID: Count):")
    for cid, count in class_counts.head(5).items():
        print(f"  Class {cid}: {count}")

    print("Bottom 5 Rare Classes (ID: Count):")
    for cid, count in class_counts.tail(5).items():
        print(f"  Class {cid}: {count}")
    print("-" * 30)
    return class_counts


def analyze_images(df):
    print("SECTION 2: INPUT DATA ANALYSIS (IMAGE SPECIFIC)")

    # Sampling
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Analysis performed on a random sample of {len(sample_df)} images.")

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Meta-features for relationship analysis
    meta_data = []

    for _, row in sample_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_name"])

        # Check existence
        if not os.path.exists(file_path):
            continue

        # Read image
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue

            # OpenCV loads as BGR, convert to RGB for stats
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels_list.append(c)

            # Pixel Stats (Normalize to 0-1 for calculation)
            img_norm = img / 255.0
            pixel_sum += np.sum(img_norm, axis=(0, 1))
            pixel_sq_sum += np.sum(img_norm**2, axis=(0, 1))
            pixel_count += h * w

            # Store for meta-analysis
            super_cat = get_super_category(row["file_name"])
            meta_data.append(
                {
                    "super_category": super_cat,
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h > 0 else 0,
                }
            )

        except Exception:
            continue

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Dimensions Analysis
    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # Channels Analysis
    c_counts = Counter(channels_list)
    print("Channel Distribution:")
    for c_num, count in c_counts.items():
        print(f"  {c_num} Channels: {count} images ({count/len(sample_df)*100:.2f}%)")

    # Pixel Stats Analysis
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        # std = sqrt(E[x^2] - (E[x])^2)
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))

        print("Pixel Statistics (RGB, Normalized 0-1):")
        print(f"  Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
        print(f"  Std : R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

    print("-" * 30)
    return pd.DataFrame(meta_data)


def analyze_relationships(meta_df):
    print("SECTION 3: FEATURE/SIGNAL RELATIONSHIPS (META-FEATURE ANALYSIS)")

    if meta_df.empty:
        print("No metadata available for relationship analysis.")
        return

    # Analyze Super Category Distribution in Sample
    print("Distribution of Super Categories (Top-level directories):")
    super_cat_counts = meta_df["super_category"].value_counts()
    for cat, count in super_cat_counts.items():
        print(f"  {cat}: {count} samples")

    print("\nRelationship: Image Dimensions by Super Category:")
    # Group by super category and calculate mean width/height/aspect ratio
    grouped = meta_df.groupby("super_category")[
        ["width", "height", "aspect_ratio"]
    ].mean()

    for cat, row in grouped.iterrows():
        print(
            f"  {cat}: Avg W={row['width']:.1f}, Avg H={row['height']:.1f}, Avg AR={row['aspect_ratio']:.4f}"
        )

    print("-" * 30)


def main():
    set_seed(RANDOM_SEED)

    # Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # 1. Target Analysis
    analyze_targets(df)

    # 2. Image Analysis
    meta_df = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
