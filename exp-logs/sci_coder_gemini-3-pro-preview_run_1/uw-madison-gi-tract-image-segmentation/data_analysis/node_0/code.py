import os
import numpy as np
import pandas as pd
import cv2
import random
from collections import Counter

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 2000  # Number of images to sample for pixel stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_area(rle_str):
    """Calculates the number of pixels in a mask from RLE string."""
    if not rle_str or rle_str == "":
        return 0
    # RLE format: start length start length ...
    # We only need the lengths (every second value)
    s = rle_str.split()
    return sum(int(x) for x in s[1::2])


def analyze_targets(df):
    print("=== TARGET VARIABLE ANALYSIS ===")

    classes = ["large_bowel", "small_bowel", "stomach"]
    total_slices = len(df)

    # 1. Distribution (Slice level)
    print("--- Class Distribution (Slice Level) ---")
    class_counts = {}
    for c in classes:
        count = (df[c] != "").sum()
        class_counts[c] = count
        print(
            f"{c.replace('_', ' ').title()}: {count} slices ({count/total_slices:.4%})"
        )

    empty_slices = (
        (df["large_bowel"] == "") & (df["small_bowel"] == "") & (df["stomach"] == "")
    ).sum()
    print(
        f"Empty Slices (Background only): {empty_slices} ({empty_slices/total_slices:.4%})"
    )

    # 2. Imbalance (Pixel level)
    print("\n--- Pixel Level Imbalance ---")
    # We estimate total pixels based on width * height
    total_pixels_dataset = (df["width"] * df["height"]).sum()

    for c in classes:
        # Vectorized RLE area calculation is tricky, looping is safer for RLE parsing
        # But to be fast, we apply a lambda
        mask_areas = df[c].apply(rle_area)
        total_mask_pixels = mask_areas.sum()
        ratio = total_mask_pixels / total_pixels_dataset
        print(
            f"{c.replace('_', ' ').title()} Ratio: {ratio:.4f} (Foreground/Total Pixels)"
        )

    # 3. Co-occurrence
    print("\n--- Class Co-occurrence ---")
    df["mask_count"] = (
        (df["large_bowel"] != "").astype(int)
        + (df["small_bowel"] != "").astype(int)
        + (df["stomach"] != "").astype(int)
    )

    counts = df["mask_count"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"Slices with {k} active classes: {v} ({v/total_slices:.4%})")


def analyze_images(df):
    print("\n=== INPUT DATA ANALYSIS (IMAGE MODALITY) ===")

    # 1. Dimensions
    print("--- Image Dimensions ---")
    widths = df["width"]
    heights = df["height"]
    aspect_ratios = widths / heights

    print(
        f"Width: Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # Unique resolutions
    resolutions = df.groupby(["width", "height"]).size().reset_index(name="count")
    print("Unique Resolutions (WxH):")
    for _, row in resolutions.iterrows():
        print(f"  {row['width']}x{row['height']}: {row['count']} images")

    # 2. Physical Spacing
    print("\n--- Physical Pixel Spacing (mm) ---")
    print(
        f"Spacing X: Mean={df['spacing_x'].mean():.4f}, Min={df['spacing_x'].min():.4f}, Max={df['spacing_x'].max():.4f}"
    )
    print(
        f"Spacing Y: Mean={df['spacing_y'].mean():.4f}, Min={df['spacing_y'].min():.4f}, Max={df['spacing_y'].max():.4f}"
    )

    # 3. Pixel Stats (Sampling)
    print("\n--- Pixel Intensity Statistics (Sampled) ---")
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    pixel_sum = 0
    pixel_sq_sum = 0
    pixel_count = 0
    min_val = float("inf")
    max_val = float("-inf")
    channels_seen = set()

    # We also want to check if image intensity correlates with presence of mask
    # So we track means separately
    mask_means = []
    no_mask_means = []

    for idx, row in sample_df.iterrows():
        img_path = os.path.join(INPUT_DIR, row["file_path"])
        # Read as is (flags=-1 usually reads as is, but cv2.IMREAD_UNCHANGED is safer)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # Check channels
        if len(img.shape) == 2:
            channels_seen.add(1)
            pixels = img.flatten()
        else:
            channels_seen.add(img.shape[2])
            pixels = img.flatten()

        # Stats accumulation
        current_mean = np.mean(pixels)
        pixel_sum += np.sum(pixels)
        pixel_sq_sum += np.sum(pixels**2)
        pixel_count += len(pixels)
        min_val = min(min_val, np.min(pixels))
        max_val = max(max_val, np.max(pixels))

        # Correlation check data
        has_mask = (
            (row["large_bowel"] != "")
            or (row["small_bowel"] != "")
            or (row["stomach"] != "")
        )
        if has_mask:
            mask_means.append(current_mean)
        else:
            no_mask_means.append(current_mean)

    global_mean = pixel_sum / pixel_count
    global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))

    print(f"Sample Size: {len(sample_df)}")
    print(f"Channels Detected: {sorted(list(channels_seen))} (1=Grayscale, 3=RGB)")
    print(f"Global Pixel Mean: {global_mean:.4f}")
    print(f"Global Pixel Std: {global_std:.4f}")
    print(f"Global Min Value: {min_val}")
    print(f"Global Max Value: {max_val}")

    return mask_means, no_mask_means


def analyze_relationships(df, mask_means, no_mask_means):
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # 1. Meta-Feature: Relative Slice Position vs Class Presence
    # We need to normalize slice ID because different scans have different numbers of slices.
    # df['slice'] is a string '0001', convert to int
    df["slice_idx"] = df["slice"].astype(int)

    # Calculate max slice per case_day
    max_slices = df.groupby(["case", "day"])["slice_idx"].max().reset_index()
    max_slices.rename(columns={"slice_idx": "max_slice"}, inplace=True)

    df = pd.merge(df, max_slices, on=["case", "day"], how="left")
    df["relative_position"] = df["slice_idx"] / df["max_slice"]

    print("--- Relative Slice Position (0.0=Top, 1.0=Bottom) vs Organ Presence ---")
    classes = ["large_bowel", "small_bowel", "stomach"]

    for c in classes:
        # Filter rows where this class is present
        present_df = df[df[c] != ""]
        if len(present_df) > 0:
            mean_pos = present_df["relative_position"].mean()
            std_pos = present_df["relative_position"].std()
            p25 = present_df["relative_position"].quantile(0.25)
            p75 = present_df["relative_position"].quantile(0.75)
            print(
                f"{c.replace('_', ' ').title()}: Mean Pos={mean_pos:.4f}, IQR=[{p25:.4f}, {p75:.4f}]"
            )
        else:
            print(f"{c.replace('_', ' ').title()}: Not present in training set")

    # 2. Image Intensity vs Mask Presence
    print("\n--- Image Intensity vs Target Presence ---")
    if mask_means and no_mask_means:
        avg_mask_intensity = np.mean(mask_means)
        avg_no_mask_intensity = np.mean(no_mask_means)
        print(f"Mean Intensity (Slices with Mask): {avg_mask_intensity:.4f}")
        print(f"Mean Intensity (Slices without Mask): {avg_no_mask_intensity:.4f}")
        diff = avg_mask_intensity - avg_no_mask_intensity
        print(f"Difference: {diff:.4f} (Positive indicates masked slices are brighter)")
    else:
        print("Insufficient data to compare intensities.")


def main():
    set_seed(SEED)

    # 1. Data Integrity
    print("=== DATA INTEGRITY ===")
    print(f"Loading metadata from {METADATA_PATH}...")
    if not os.path.exists(METADATA_PATH):
        print("Error: Metadata file not found.")
        return

    # keep_default_na=False ensures empty strings are not NaN
    df = pd.read_csv(METADATA_PATH, keep_default_na=False)
    print(f"Loaded {len(df)} training samples.")
    print("Confirmed: Analysis restricted to training set only.")
    print("")

    # 2. Target Analysis
    analyze_targets(df)

    # 3. Image Analysis
    # We pass the df to get stats, and it returns lists for relationship analysis
    mask_means, no_mask_means = analyze_images(df)

    # 4. Relationships
    analyze_relationships(df, mask_means, no_mask_means)


if __name__ == "__main__":
    main()
