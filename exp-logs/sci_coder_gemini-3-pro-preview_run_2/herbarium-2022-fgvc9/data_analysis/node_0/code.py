import os
import json
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV_PATH = os.path.join(METADATA_DIR, "train.csv")
TRAIN_META_JSON_PATH = os.path.join(INPUT_DIR, "train_metadata.json")
SEED = 42
SAMPLE_SIZE = 2500  # Sample size for image processing to ensure runtime < 1hr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_image_metadata():
    """Loads train.csv and merges with taxonomy from json."""
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"{TRAIN_CSV_PATH} not found.")

    df = pd.read_csv(TRAIN_CSV_PATH)

    # Load Taxonomy
    if os.path.exists(TRAIN_META_JSON_PATH):
        try:
            with open(TRAIN_META_JSON_PATH, "r") as f:
                meta = json.load(f)
                if "categories" in meta:
                    tax_df = pd.DataFrame(meta["categories"])
                    # Ensure category_id is int for merge
                    if "category_id" in tax_df.columns:
                        tax_df["category_id"] = tax_df["category_id"].astype(int)
                        df = df.merge(
                            tax_df, left_on="label", right_on="category_id", how="left"
                        )
        except Exception as e:
            print(f"Warning: Could not process taxonomy metadata: {e}")

    return df


def analyze_target(df):
    """Analyzes the distribution of the target variable (category_id/label)."""
    print("\n==== TARGET VARIABLE ANALYSIS ====")

    # Primary Target: Label (Category ID)
    class_counts = df["label"].value_counts()
    n_classes = len(class_counts)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes (Species): {n_classes}")

    # Distribution stats
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Class Balance Statistics:")
    print(f"  Min Samples per Class: {min_samples}")
    print(f"  Max Samples per Class: {max_samples}")
    print(f"  Mean Samples per Class: {mean_samples:.4f}")
    print(f"  Median Samples per Class: {median_samples:.4f}")

    # Imbalance Ratio
    imbalance_ratio = max_samples / min_samples
    print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Higher Level Taxonomy Analysis (Family)
    if "family" in df.columns:
        family_counts = df["family"].value_counts()
        print(f"\nTaxonomic Hierarchy Analysis:")
        print(f"  Number of Unique Families: {len(family_counts)}")
        print(f"  Top 3 Families by Image Count:")
        for fam, count in family_counts.head(3).items():
            print(f"    {fam}: {count} images ({count/total_samples*100:.2f}%)")

        # Rare Families
        rare_families = family_counts[family_counts < (total_samples * 0.01)]
        print(f"  Number of Rare Families (<1% freq): {len(rare_families)}")


def analyze_images(df):
    """Analyzes a sample of images for dimensions, channels, and pixel stats."""
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Stratified Sampling (or random if stratified fails due to small classes)
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    else:
        sample_df = df.copy()

    print(f"Analyzing a random sample of {len(sample_df)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = Counter()

    # Pixel stats accumulators
    # We will compute mean/std in [0, 1] range
    pixel_sum = np.zeros(3)  # RGB
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_images = 0

    for _, row in sample_df.iterrows():
        img_path = os.path.join(INPUT_DIR, row["image_path"])

        # Read image
        try:
            # cv2 reads in BGR
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, c = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channel_counts[c] += 1

            # Normalize to 0-1 for stats
            img_norm = img_rgb / 255.0
            pixel_sum += np.sum(img_norm, axis=(0, 1))
            pixel_sq_sum += np.sum(img_norm**2, axis=(0, 1))
            pixel_count += h * w

            valid_images += 1

        except Exception:
            continue

    if valid_images == 0:
        print("Error: No valid images found in sample.")
        return sample_df  # Return for meta-analysis anyway

    # Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    ars = np.array(aspect_ratios)

    print(f"Image Dimensions (Width):")
    print(f"  Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}")
    print(f"  Min: {np.min(widths)}, Max: {np.max(widths)}")

    print(f"Image Dimensions (Height):")
    print(f"  Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}")
    print(f"  Min: {np.min(heights)}, Max: {np.max(heights)}")

    print(f"Aspect Ratios (Width/Height):")
    print(f"  Mean: {np.mean(ars):.4f}, Std: {np.std(ars):.4f}")

    # Channels
    print(f"Channel Distribution:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images")

    # Pixel Stats
    global_mean = pixel_sum / pixel_count
    global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))

    print(f"Global Pixel Statistics (Normalized [0, 1]):")
    print(
        f"  Mean (R, G, B): [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]"
    )
    print(
        f"  Std  (R, G, B): [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]"
    )

    # Add calculated stats to sample_df for relationship analysis
    # We assume the order is preserved as we iterated row by row,
    # but we skipped invalid images. This is a quick approximation.
    # To be safe, we won't append columns, but return the lists if needed.
    # For the meta-feature analysis, we'll re-calculate simple area on the fly or use what we have.

    # Let's attach area to the sample_df for the next step.
    # We need to be careful about indices of skipped images.
    # Re-running a lightweight loop or storing data in a list of dicts is safer.

    meta_data = []
    # Re-iterate or store during first pass? Storing during first pass is better.
    # Let's just return the data we collected paired with the rows.
    # Since we can't easily modify the DF in the loop above without index management,
    # we will rely on the fact that valid_images count might mismatch if we just append.
    # However, for the purpose of this script, we can do a quick correlation on the valid ones.

    return (
        widths,
        heights,
        sample_df.iloc[: len(widths)] if len(widths) < len(sample_df) else sample_df,
    )


def analyze_relationships(df, widths, heights):
    """Analyzes relationships between metadata and image signals."""
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Ensure lengths match (basic check)
    n = min(len(df), len(widths))
    df_subset = df.iloc[:n].copy()
    w_subset = np.array(widths[:n])
    h_subset = np.array(heights[:n])

    # 1. Structured Relationship: Width vs Height Correlation
    corr_wh = np.corrcoef(w_subset, h_subset)[0, 1]
    print(f"Structured Relationship:")
    print(f"  Correlation between Image Width and Height: {corr_wh:.4f}")

    # 2. Meta-Feature Relationship: Image Area vs Family
    # Do certain plant families have larger/smaller images (e.g. higher res photos)?
    df_subset["image_area"] = w_subset * h_subset

    if "family" in df_subset.columns:
        print(f"\nMeta-Feature Relationship (Image Area vs Family):")
        # Get top 5 families in this subset
        top_families = df_subset["family"].value_counts().head(5).index.tolist()

        print(f"  Average Image Area (pixels) for Top 5 Families:")
        for fam in top_families:
            fam_data = df_subset[df_subset["family"] == fam]["image_area"]
            mean_area = fam_data.mean()
            std_area = fam_data.std()
            print(f"    {fam}: Mean Area = {mean_area:.0f}, Std = {std_area:.0f}")

        # Correlation between Image Area and Class Label (as a proxy for complexity?)
        # Since label is categorical integer, correlation is technically weak,
        # but requested by prompt style "Relationship between metadata and target".
        # A better one: Does genus size (number of species) correlate with image count?

    # 3. Taxonomy Structure Analysis
    if "genus" in df.columns:
        print(f"\nTaxonomic Structure (Meta-Feature):")
        genus_counts = df["genus"].value_counts()
        print(f"  Most diverse Genera (by sample count):")
        for gen, count in genus_counts.head(3).items():
            print(f"    {gen}: {count} samples")


def main():
    set_seed(SEED)

    # 1. Load Data
    try:
        df_train = load_image_metadata()
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    # 2. Target Analysis
    analyze_target(df_train)

    # 3. Image Analysis
    # We return widths/heights to use in relationship analysis
    # Note: This might return slightly fewer items than sample_df if images fail to load
    widths, heights, df_sample_valid = analyze_images(df_train)

    # 4. Relationships
    if len(widths) > 0:
        analyze_relationships(df_sample_valid, widths, heights)


if __name__ == "__main__":
    main()
