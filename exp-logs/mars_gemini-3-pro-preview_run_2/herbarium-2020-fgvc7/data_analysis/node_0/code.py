import os
import pandas as pd
import numpy as np
import cv2
import random
import concurrent.futures
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def get_image_stats(file_info):
    """
    Helper function to process a single image.
    Args:
        file_info: tuple of (full_path, category_id, region_id)
    Returns:
        dict with stats or None if failed
    """
    path, cat_id, reg_id = file_info

    try:
        # Read image
        img = cv2.imread(path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        # Calculate pixel stats (mean and std per channel)
        # To save time, we calculate these on the individual image
        # Global stats will be averaged later
        mean_pixel = img.mean(axis=(0, 1))
        std_pixel = img.std(axis=(0, 1))

        return {
            "width": w,
            "height": h,
            "channels": c,
            "aspect_ratio": w / h if h > 0 else 0,
            "mean_r": mean_pixel[0],
            "mean_g": mean_pixel[1],
            "mean_b": mean_pixel[2],
            "std_r": std_pixel[0],
            "std_g": std_pixel[1],
            "std_b": std_pixel[2],
            "category_id": cat_id,
            "region_id": reg_id,
        }
    except Exception:
        return None


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")

    print("Loading training metadata...")
    df = pd.read_csv(TRAIN_CSV)

    # ---------------------------------------------------------
    # 1. Target Variable Analysis
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "category_id"
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Distribution stats
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Class Balance Statistics:")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Mean samples per class: {mean_samples:.4f}")
    print(f"  Median samples per class: {median_samples:.4f}")

    # Top/Bottom classes
    print("\nTop 5 Most Frequent Classes:")
    for cls, count in class_counts.head(5).items():
        ratio = count / total_samples
        print(f"  Class {cls}: {count} samples ({ratio:.4%})")

    print("\nTop 5 Least Frequent Classes:")
    for cls, count in class_counts.tail(5).items():
        ratio = count / total_samples
        print(f"  Class {cls}: {count} samples ({ratio:.4%})")

    # Imbalance check
    imbalance_ratio = max_samples / min_samples
    print(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # ---------------------------------------------------------
    # 2. Input Data Analysis (Image Data)
    # ---------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Sample images for analysis to keep runtime within limits
    SAMPLE_SIZE = 2000
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        sample_df = df.copy()

    print(f"Sampling {len(sample_df)} images for detailed pixel/dimension analysis...")

    # Prepare paths
    # The file_path in csv is relative to input dir, e.g., "nybg2020/train/..."
    # We need to prepend INPUT_DIR
    paths = [os.path.join(INPUT_DIR, row.file_path) for row in sample_df.itertuples()]
    cats = sample_df["category_id"].tolist()
    regs = sample_df["region_id"].tolist()

    tasks = list(zip(paths, cats, regs))

    image_stats = []

    # Use ThreadPoolExecutor for I/O bound task
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(get_image_stats, tasks))

    # Filter out Nones
    image_stats = [r for r in results if r is not None]

    if not image_stats:
        print("Error: No images could be processed.")
        return

    stats_df = pd.DataFrame(image_stats)

    # Dimensions
    print("\nDimensions:")
    print(
        f"  Width:  Mean={stats_df['width'].mean():.4f}, Std={stats_df['width'].std():.4f}, Min={stats_df['width'].min()}, Max={stats_df['width'].max()}"
    )
    print(
        f"  Height: Mean={stats_df['height'].mean():.4f}, Std={stats_df['height'].std():.4f}, Min={stats_df['height'].min()}, Max={stats_df['height'].max()}"
    )

    # Aspect Ratio
    print("\nAspect Ratios (Width/Height):")
    print(f"  Mean: {stats_df['aspect_ratio'].mean():.4f}")
    print(f"  Std:  {stats_df['aspect_ratio'].std():.4f}")
    print(f"  Min:  {stats_df['aspect_ratio'].min():.4f}")
    print(f"  Max:  {stats_df['aspect_ratio'].max():.4f}")

    # Channels
    channel_counts = stats_df["channels"].value_counts()
    print("\nChannel Distribution:")
    for ch, count in channel_counts.items():
        print(f"  {ch} Channels: {count} images ({count/len(stats_df):.2%})")

    # Pixel Stats (Global approximation from sample means)
    # Note: Averaging per-image means is a valid approximation for global mean if image sizes are roughly similar.
    print("\nPixel Value Statistics (RGB, 0-255):")
    print(f"  Mean R: {stats_df['mean_r'].mean():.4f}")
    print(f"  Mean G: {stats_df['mean_g'].mean():.4f}")
    print(f"  Mean B: {stats_df['mean_b'].mean():.4f}")
    print(f"  Std R:  {stats_df['std_r'].mean():.4f} (Avg of per-image std)")
    print(f"  Std G:  {stats_df['std_g'].mean():.4f} (Avg of per-image std)")
    print(f"  Std B:  {stats_df['std_b'].mean():.4f} (Avg of per-image std)")

    # ---------------------------------------------------------
    # 3. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Structured Relationships: Metadata vs Target
    # Relationship between Region and Category
    # Check if categories are region-specific

    print("Metadata Analysis (Region vs Category):")

    # Group categories by region
    cat_per_region = df.groupby("region_id")["category_id"].nunique()
    print("\nUnique Categories per Region:")
    for reg, count in cat_per_region.items():
        print(f"  Region {reg}: {count} unique categories")

    # Check overlap: How many categories appear in multiple regions?
    cat_region_counts = df.groupby("category_id")["region_id"].nunique()
    multi_region_cats = (cat_region_counts > 1).sum()
    single_region_cats = (cat_region_counts == 1).sum()

    print(f"\nCategory Geographic Specificity:")
    print(
        f"  Categories found in only 1 region: {single_region_cats} ({single_region_cats/num_classes:.2%})"
    )
    print(
        f"  Categories found in >1 regions:    {multi_region_cats} ({multi_region_cats/num_classes:.2%})"
    )

    # Unstructured Relationships: Image Size vs Category
    # Do certain categories tend to have larger/smaller images?
    # We use the sampled data for this.

    # Calculate image area
    stats_df["area"] = stats_df["width"] * stats_df["height"]

    # Correlation between Area and Category ID (Numerical correlation is meaningless for nominal ID,
    # but we can check if there's variance in size across categories)

    # Let's look at the top 5 most frequent categories in our sample and their average image area
    top_sample_cats = stats_df["category_id"].value_counts().head(5).index

    print("\nImage Area vs Top 5 Categories (in sample):")
    for cat in top_sample_cats:
        cat_stats = stats_df[stats_df["category_id"] == cat]
        mean_area = cat_stats["area"].mean()
        std_area = cat_stats["area"].std()
        n_samples = len(cat_stats)
        print(
            f"  Class {cat} (n={n_samples}): Mean Area = {mean_area:.0f} px², Std = {std_area:.0f}"
        )

    # Check correlation between Aspect Ratio and Region
    # Maybe different regions use different camera setups/standards?
    print("\nAspect Ratio vs Region (in sample):")
    regions_in_sample = stats_df["region_id"].unique()
    for reg in sorted(regions_in_sample):
        reg_stats = stats_df[stats_df["region_id"] == reg]
        mean_ar = reg_stats["aspect_ratio"].mean()
        print(f"  Region {reg}: Mean Aspect Ratio = {mean_ar:.4f}")


if __name__ == "__main__":
    main()
