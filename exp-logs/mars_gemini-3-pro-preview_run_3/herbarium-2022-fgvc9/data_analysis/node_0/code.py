import os
import json
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
TRAIN_METADATA_JSON = "./input/train_metadata.json"
SEED = 42
SAMPLE_SIZE = 2000  # Number of images to sample for pixel stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file {METADATA_FILE} not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    # ---------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Target is category_id
    target_col = "category_id"
    class_counts = df[target_col].value_counts()

    total_samples = len(df)
    num_classes = len(class_counts)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Imbalance
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()
    std_samples = class_counts.std()

    print(f"Class Balance Statistics:")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Mean samples per class: {mean_samples:.4f}")
    print(f"  Median samples per class: {median_samples:.4f}")
    print(f"  Std samples per class: {std_samples:.4f}")

    # Top/Bottom classes
    print(f"Top 5 Most Common Classes (ID: Count):")
    for cid, count in class_counts.head(5).items():
        print(f"  Class {cid}: {count}")

    print(f"Bottom 5 Least Common Classes (ID: Count):")
    for cid, count in class_counts.tail(5).items():
        print(f"  Class {cid}: {count}")

    # Hierarchy Analysis
    try:
        if os.path.exists(TRAIN_METADATA_JSON):
            # Attempt to load JSON to extract hierarchy
            with open(TRAIN_METADATA_JSON, "r") as f:
                meta_json = json.load(f)

            if "categories" in meta_json:
                cats = meta_json["categories"]
                cat_df = pd.DataFrame(cats)

                # Check for hierarchy columns
                if "family" in cat_df.columns and "genus" in cat_df.columns:
                    num_families = cat_df["family"].nunique()
                    num_genera = cat_df["genus"].nunique()

                    print(f"Taxonomic Hierarchy Analysis:")
                    print(f"  Unique Families: {num_families}")
                    print(f"  Unique Genera: {num_genera}")

                    # Merge to get distribution
                    if "id" in cat_df.columns:
                        cat_df_renamed = cat_df.rename(columns={"id": "category_id"})
                        # Merge with training data to see distribution in the actual dataset
                        merged_df = df.merge(
                            cat_df_renamed[["category_id", "family", "genus"]],
                            on="category_id",
                            how="left",
                        )

                        top_families = merged_df["family"].value_counts().head(5)
                        print(f"  Top 5 Families by Sample Count:")
                        for fam, count in top_families.items():
                            print(f"    {fam}: {count}")
    except Exception as e:
        print(f"  (Hierarchy analysis skipped: {str(e)})")

    # ---------------------------------------------------------
    # SECTION 2: INPUT DATA ANALYSIS (IMAGE)
    # ---------------------------------------------------------
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Sample images
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    # Accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_stats = []

    print(f"Analyzing {len(sample_df)} sampled images...")

    for idx, row in sample_df.iterrows():
        path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            if not os.path.exists(path):
                continue

            fsize = os.path.getsize(path)

            # Read image
            img = cv2.imread(path)
            if img is None:
                continue

            h, w, c = img.shape

            # Pixel stats
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3) / 255.0

            pixel_sum += pixels.sum(axis=0)
            pixel_sq_sum += (pixels**2).sum(axis=0)
            pixel_count += pixels.shape[0]

            valid_stats.append(
                {
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h > 0 else 0,
                    "channels": c,
                    "file_size": fsize,
                }
            )

        except Exception:
            continue

    if not valid_stats:
        print("Error: No valid images processed.")
        return

    stats_df = pd.DataFrame(valid_stats)

    # Dimensions
    print(f"Image Dimensions:")
    print(
        f"  Width:  Mean={stats_df['width'].mean():.4f}, Std={stats_df['width'].std():.4f}, Min={stats_df['width'].min()}, Max={stats_df['width'].max()}"
    )
    print(
        f"  Height: Mean={stats_df['height'].mean():.4f}, Std={stats_df['height'].std():.4f}, Min={stats_df['height'].min()}, Max={stats_df['height'].max()}"
    )
    print(
        f"  Aspect Ratio: Mean={stats_df['aspect_ratio'].mean():.4f}, Std={stats_df['aspect_ratio'].std():.4f}"
    )

    # Channels
    c_counts = stats_df["channels"].value_counts().to_dict()
    print(f"Channel Counts: {c_counts}")

    # Global Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))
        print(f"Pixel Statistics (Normalized 0-1):")
        print(
            f"  Mean (R, G, B): [{rgb_mean[0]:.4f}, {rgb_mean[1]:.4f}, {rgb_mean[2]:.4f}]"
        )
        print(
            f"  Std  (R, G, B): [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]"
        )

    # ---------------------------------------------------------
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Correlation between Width and Height
    if len(stats_df) > 1:
        corr_wh = stats_df["width"].corr(stats_df["height"])
        print(f"Correlation between Width and Height: {corr_wh:.4f}")

        # Correlation between File Size and Resolution (W*H)
        resolutions = stats_df["width"] * stats_df["height"]
        corr_size_res = stats_df["file_size"].corr(resolutions)
        print(
            f"Correlation between File Size and Resolution (Pixels): {corr_size_res:.4f}"
        )

        # Correlation between Aspect Ratio and File Size
        corr_ar_size = stats_df["aspect_ratio"].corr(stats_df["file_size"])
        print(f"Correlation between Aspect Ratio and File Size: {corr_ar_size:.4f}")


if __name__ == "__main__":
    main()
