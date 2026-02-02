import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy import stats

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
RANDOM_SEED = 42
SAMPLE_SIZE_PIXELS = 1000  # Number of images to sample for pixel statistics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rle_area(rle_string):
    """Calculates the total number of masked pixels from an RLE string."""
    if pd.isna(rle_string) or rle_string == "":
        return 0
    # RLE format: start length start length ...
    # We only need to sum the lengths (every second value)
    values = [int(x) for x in rle_string.split()]
    return sum(values[1::2])


def analyze_targets(df):
    print("=== TARGET VARIABLE ANALYSIS ===")

    classes = ["large_bowel", "small_bowel", "stomach"]
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")

    # 1. Distribution (Presence of masks)
    print("\n--- Class Distribution (Mask Presence) ---")
    for cls in classes:
        # Count non-empty masks
        count = df[cls].apply(lambda x: 0 if pd.isna(x) or x == "" else 1).sum()
        ratio = count / total_samples
        print(f"{cls}: {count} samples ({ratio*100:.2f}%)")

    # Check for empty samples (background only)
    # A sample is empty if all 3 class columns are empty
    df["has_mask"] = df[classes].apply(
        lambda row: any([x != "" and not pd.isna(x) for x in row]), axis=1
    )
    empty_samples = total_samples - df["has_mask"].sum()
    print(
        f"Background only (no masks): {empty_samples} samples ({(empty_samples/total_samples)*100:.2f}%)"
    )

    # 2. Mask Area Analysis (Imbalance in size)
    print("\n--- Mask Area Analysis (Pixels) ---")
    for cls in classes:
        # Calculate area for non-empty masks only
        areas = df[df[cls] != ""][cls].apply(rle_area)
        if len(areas) > 0:
            print(f"{cls} Area per Slice:")
            print(f"  Mean: {areas.mean():.4f}")
            print(f"  Std : {areas.std():.4f}")
            print(f"  Min : {areas.min()}")
            print(f"  Max : {areas.max()}")
        else:
            print(f"{cls}: No masks found.")


def analyze_images(df):
    print("\n=== INPUT DATA ANALYSIS (IMAGE) ===")

    # 1. Dimensions
    print("\n--- Image Dimensions ---")
    print("Height stats:")
    print(df["height"].describe().to_string())
    print("\nWidth stats:")
    print(df["width"].describe().to_string())

    # Aspect Ratios
    aspect_ratios = df["width"] / df["height"]
    print(f"\nAspect Ratio Mean: {aspect_ratios.mean():.4f}")
    print(f"Aspect Ratio Std : {aspect_ratios.std():.4f}")
    unique_dims = df.groupby(["width", "height"]).size().reset_index(name="count")
    print(f"\nUnique Dimensions (WxH):")
    for _, row in unique_dims.iterrows():
        print(f"  {row['width']}x{row['height']}: {row['count']} images")

    # 2. Pixel Spacing
    print("\n--- Physical Pixel Spacing (mm) ---")
    print(f"Pixel Spacing Height Mean: {df['pixel_spacing_h'].mean():.4f}")
    print(f"Pixel Spacing Width Mean : {df['pixel_spacing_w'].mean():.4f}")

    # 3. Pixel Statistics (Sampling)
    print(f"\n--- Pixel Intensity Stats (Sampled n={SAMPLE_SIZE_PIXELS}) ---")

    sample_df = df.sample(n=min(len(df), SAMPLE_SIZE_PIXELS), random_state=RANDOM_SEED)

    pixel_means = []
    pixel_stds = []
    pixel_mins = []
    pixel_maxs = []
    channels = []

    for _, row in sample_df.iterrows():
        path = os.path.join(INPUT_DIR, row["image_path"])
        if os.path.exists(path):
            # Load as is (likely grayscale or uint16)
            # Note: cv2.imread loads as BGR (3 channels) by default if flags not set,
            # but medical images might be 16-bit. The prompt mentions .png, likely 8-bit or 16-bit.
            # We use IMREAD_UNCHANGED to detect true depth/channels.
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            if len(img.shape) == 2:
                c = 1
            else:
                c = img.shape[2]
            channels.append(c)

            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))
            pixel_mins.append(np.min(img))
            pixel_maxs.append(np.max(img))

    if channels:
        unique_channels, counts = np.unique(channels, return_counts=True)
        print("Channel Distribution:")
        for c, count in zip(unique_channels, counts):
            print(f"  {c} channel(s): {count} images")

        print(f"Global Pixel Mean (Sampled): {np.mean(pixel_means):.4f}")
        print(
            f"Global Pixel Std  (Sampled): {np.mean(pixel_stds):.4f}"
        )  # Average of stds is a rough approx
        print(f"Pixel Value Min (Sampled): {np.min(pixel_mins)}")
        print(f"Pixel Value Max (Sampled): {np.max(pixel_maxs)}")
    else:
        print("Could not load images for pixel analysis.")


def analyze_relationships(df):
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    classes = ["large_bowel", "small_bowel", "stomach"]

    # 1. Co-occurrence
    print("\n--- Class Co-occurrence ---")
    # Create binary columns for presence
    for cls in classes:
        df[f"has_{cls}"] = df[cls].apply(
            lambda x: 1 if x != "" and not pd.isna(x) else 0
        )

    co_matrix = df[[f"has_{c}" for c in classes]].corr()
    print("Correlation Matrix (Presence):")
    print(co_matrix.round(4).to_string())

    # 2. Slice Position Analysis
    # Normalize slice numbers per case/day to 0-1 range to see where organs appear
    print("\n--- Organ Location vs Slice Depth ---")

    # Ensure slice is int
    df["slice"] = df["slice"].astype(int)

    # Calculate max slice per scan group
    scan_group = (
        df.groupby(["case", "day"])["slice"]
        .max()
        .reset_index()
        .rename(columns={"slice": "max_slice"})
    )
    df = df.merge(scan_group, on=["case", "day"], how="left")

    # Normalized position (0 = top, 1 = bottom)
    df["rel_position"] = df["slice"] / df["max_slice"]

    for cls in classes:
        subset = df[df[f"has_{cls}"] == 1]
        if not subset.empty:
            mean_pos = subset["rel_position"].mean()
            std_pos = subset["rel_position"].std()
            print(
                f"{cls} Relative Position (0.0-1.0): Mean={mean_pos:.4f}, Std={std_pos:.4f}"
            )
        else:
            print(f"{cls}: No occurrences.")

    # 3. Metadata Correlations
    # Check if image size correlates with mask area (e.g., larger images have larger organs?)
    # We sum areas of all organs for this check
    df["total_mask_area"] = df[classes].apply(
        lambda row: sum([rle_area(x) for x in row]), axis=1
    )

    # Correlation between image height and mask area
    corr_h = df["height"].corr(df["total_mask_area"])
    corr_w = df["width"].corr(df["total_mask_area"])

    print("\n--- Metadata vs Target Correlations ---")
    print(f"Image Height vs Total Mask Area Correlation: {corr_h:.4f}")
    print(f"Image Width  vs Total Mask Area Correlation: {corr_w:.4f}")


def main():
    set_seed(RANDOM_SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    try:
        df = pd.read_csv(METADATA_PATH)
        # Fill NaNs in segmentation columns with empty strings
        df[["large_bowel", "small_bowel", "stomach"]] = df[
            ["large_bowel", "small_bowel", "stomach"]
        ].fillna("")

        analyze_targets(df)
        analyze_images(df)
        analyze_relationships(df)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
