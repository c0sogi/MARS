import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter
from scipy.stats import skew, kurtosis

# --- Configuration ---
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE_IMG_STATS = 1500  # Number of images to sample for pixel stats/dimensions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    target_col = "category_id"

    # Distribution
    counts = df[target_col].value_counts()
    total_samples = len(df)
    num_classes = len(counts)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Top 5 Classes
    print("Top 5 Classes (ID: Count, Percentage):")
    for cat_id, count in counts.head(5).items():
        print(f"  Class {cat_id}: {count} ({count/total_samples*100:.4f}%)")

    # Imbalance
    max_class_count = counts.max()
    min_class_count = counts.min()
    imbalance_ratio = (
        max_class_count / min_class_count if min_class_count > 0 else float("inf")
    )

    print(f"Most Frequent Class Count: {max_class_count}")
    print(f"Least Frequent Class Count: {min_class_count}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Check for Class 0 (Empty) specifically
    if 0 in counts.index:
        empty_count = counts[0]
        print(
            f"Empty Images (Class 0): {empty_count} ({empty_count/total_samples*100:.4f}%)"
        )
    else:
        print("Empty Images (Class 0): 0 (0.0000%)")
    print("-" * 30)


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")

    # Sample images to save time
    if len(df) > SAMPLE_SIZE_IMG_STATS:
        # Stratified sampling if possible, otherwise random
        try:
            sample_df = df.groupby("category_id", group_keys=False).apply(
                lambda x: x.sample(
                    min(
                        len(x),
                        max(
                            1, int(SAMPLE_SIZE_IMG_STATS / df["category_id"].nunique())
                        ),
                    )
                )
            )
            # If stratified sample is too small (due to many rare classes), fill up with random
            if len(sample_df) < SAMPLE_SIZE_IMG_STATS // 2:
                sample_df = df.sample(n=SAMPLE_SIZE_IMG_STATS, random_state=SEED)
        except:
            sample_df = df.sample(n=SAMPLE_SIZE_IMG_STATS, random_state=SEED)
    else:
        sample_df = df

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_samples = 0

    print(f"Analyzing a sample of {len(sample_df)} images...")

    for _, row in sample_df.iterrows():
        # Construct full path. file_path in metadata is relative to input dir
        # Note: metadata file_path might already include 'train/', check input structure
        # Based on metadata generation script: file_path is like "train/id.jpg"
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(img_path):
            continue

        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channels_list.append(c)

            # Update pixel stats
            # Normalize to 0-1 for stats calculation to avoid overflow, then scale back or report as is
            # Here we report 0-255 stats as is common, using float64 accumulators
            img_flat = img.reshape(-1, 3)
            pixel_sum += img_flat.sum(axis=0)
            pixel_sq_sum += (img_flat**2).sum(axis=0)
            pixel_count += h * w

            valid_samples += 1

        except Exception:
            continue

    if valid_samples == 0:
        print("No valid images found in sample.")
        return

    # Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(
        f"Image Widths: Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Image Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    chan_counts = Counter(channels_list)
    print(f"Channel Distribution: {dict(chan_counts)}")

    # Pixel Stats
    # Global Mean = Sum / N
    # Global Std = Sqrt( (SumSq / N) - (Mean^2) )
    global_mean = pixel_sum / pixel_count
    global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))

    print(
        f"Pixel Mean (RGB): R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    print(
        f"Pixel Std (RGB):  R={global_std[0]:.4f},  G={global_std[1]:.4f},  B={global_std[2]:.4f}"
    )
    print("-" * 30)


def analyze_meta_relationships(df):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. MegaDetector Confidence vs Target
    # Check if 'max_detection_conf' separates Empty (0) vs Animals (>0)
    if "max_detection_conf" in df.columns:
        empty_mask = df["category_id"] == 0
        animal_mask = df["category_id"] != 0

        conf_empty = df.loc[empty_mask, "max_detection_conf"]
        conf_animal = df.loc[animal_mask, "max_detection_conf"]

        print("MegaDetector Confidence (max_detection_conf) Stats:")
        if len(conf_empty) > 0:
            print(
                f"  Class 0 (Empty)  - Mean: {conf_empty.mean():.4f}, Std: {conf_empty.std():.4f}, Median: {conf_empty.median():.4f}"
            )
        else:
            print("  Class 0 (Empty)  - No samples")

        if len(conf_animal) > 0:
            print(
                f"  Class >0 (Animal) - Mean: {conf_animal.mean():.4f}, Std: {conf_animal.std():.4f}, Median: {conf_animal.median():.4f}"
            )
        else:
            print("  Class >0 (Animal) - No samples")

        # Point Biserial Correlation (Binary: Empty vs Animal)
        # Create binary target: 0 for Empty, 1 for Animal
        binary_target = animal_mask.astype(int)
        corr = binary_target.corr(df["max_detection_conf"])
        print(f"  Correlation (Binary Target vs Confidence): {corr:.4f}")

    # 2. Location vs Target
    # Check if classes are location-specific (Cardinality of locations)
    if "location" in df.columns:
        num_locations = df["location"].nunique()
        print(f"\nLocation Analysis:")
        print(f"  Total Unique Locations: {num_locations}")

        # Calculate Mutual Information proxy or just simple overlap
        # Let's check average number of classes per location
        classes_per_loc = df.groupby("location")["category_id"].nunique()
        print(f"  Avg Classes per Location: {classes_per_loc.mean():.4f}")
        print(f"  Max Classes in one Location: {classes_per_loc.max()}")
        print(f"  Min Classes in one Location: {classes_per_loc.min()}")

    # 3. Missing Values in Metadata
    print("\nMissing Values in Metadata:")
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            print(f"  {col}: {val} ({val/len(df)*100:.4f}%)")
        else:
            pass  # Silent if no missing
    if missing.sum() == 0:
        print("  No missing values found in metadata columns.")

    print("-" * 30)


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    analyze_target(df)
    analyze_images(df)
    analyze_meta_relationships(df)


if __name__ == "__main__":
    main()
