import os
import numpy as np
import pandas as pd
import cv2
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train.csv"
INPUT_ROOT = "./input"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_images(df):
    """
    Iterates through images to calculate dimension stats, channel stats,
    and global pixel mean/std.
    """
    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # For global pixel stats (using Welford's online algorithm or simple sum accumulation)
    # Given dataset size (2400 * 101 * 101), simple sum accumulation fits in float64.
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    # We will check a subset or all. 2400 is small enough to check all.
    for idx, row in df.iterrows():
        # Construct path. The metadata contains relative path 'train/images/id.png'
        # Input root is './input'
        img_path = os.path.join(INPUT_ROOT, row["image_path"])

        # Read image
        # IMREAD_UNCHANGED to detect if it's 1 channel or 3
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        h, w = img.shape[:2]
        if len(img.shape) == 2:
            c = 1
        else:
            c = img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channel_counts.append(c)

        # Pixel stats
        # Normalize to 0-1 for calculation or keep 0-255?
        # Usually for EDA on 8-bit images, reporting in 0-255 is standard unless specified.
        # We will report in 0-255 scale.
        flat_pixels = img.flatten().astype(np.float64)
        pixel_sum += np.sum(flat_pixels)
        pixel_sq_sum += np.sum(flat_pixels**2)
        total_pixels += len(flat_pixels)

    # Calculate global stats
    global_mean = pixel_sum / total_pixels
    # Var = E[X^2] - (E[X])^2
    global_var = (pixel_sq_sum / total_pixels) - (global_mean**2)
    global_std = np.sqrt(global_var)

    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "aspect_ratios": np.array(aspect_ratios),
        "channels": np.array(channel_counts),
        "pixel_mean": global_mean,
        "pixel_std": global_std,
    }


def main():
    set_seed(SEED)

    # 1. Data Integrity & Loading
    if not os.path.exists(METADATA_PATH):
        print("Error: Metadata file not found.")
        return

    df = pd.read_csv(METADATA_PATH)

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # 2. Target Variable Analysis
    # Target is salt_coverage (derived from rle_mask)
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 24)

    coverage = df["salt_coverage"]

    # Distribution
    print(f"Target Variable: Salt Coverage (Proportion of Image)")
    print(f"Count: {len(coverage)}")
    print(f"Mean: {coverage.mean():.4f}")
    print(f"Std Dev: {coverage.std():.4f}")
    print(f"Min: {coverage.min():.4f}")
    print(f"Max: {coverage.max():.4f}")

    # Imbalance (Empty vs Non-Empty)
    # In segmentation, 'Empty' means no mask (coverage == 0)
    empty_count = (coverage == 0).sum()
    non_empty_count = (coverage > 0).sum()
    empty_ratio = empty_count / len(coverage)

    print(f"Empty Masks (No Salt): {empty_count} ({empty_ratio*100:.2f}%)")
    print(f"Non-Empty Masks: {non_empty_count} ({(1-empty_ratio)*100:.2f}%)")

    # Skewness/Kurtosis of the non-zero distribution
    if non_empty_count > 0:
        non_zero_cov = coverage[coverage > 0]
        print(f"Skewness (Non-Zero Targets): {non_zero_cov.skew():.4f}")
        print(f"Kurtosis (Non-Zero Targets): {non_zero_cov.kurtosis():.4f}")

    # 3. Input Data Analysis (Image Data)
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 27)

    img_stats = analyze_images(df)

    # Dimensions
    w_mean = img_stats["widths"].mean()
    h_mean = img_stats["heights"].mean()
    ar_mean = img_stats["aspect_ratios"].mean()

    print(f"Average Width: {w_mean:.4f}")
    print(f"Average Height: {h_mean:.4f}")
    print(f"Average Aspect Ratio: {ar_mean:.4f}")

    # Check for dimension consistency
    unique_w = np.unique(img_stats["widths"])
    unique_h = np.unique(img_stats["heights"])
    if len(unique_w) == 1 and len(unique_h) == 1:
        print(f"Dimension Consistency: All images are {unique_w[0]}x{unique_h[0]}")
    else:
        print(f"Dimension Consistency: Mixed dimensions detected.")

    # Channels
    unique_c, counts_c = np.unique(img_stats["channels"], return_counts=True)
    print(f"Channel Distribution:")
    for c, count in zip(unique_c, counts_c):
        mode = "Grayscale" if c == 1 else "RGB/Multi"
        print(f"  {c} Channel(s) ({mode}): {count} images")

    # Pixel Stats
    print(f"Global Pixel Mean: {img_stats['pixel_mean']:.4f}")
    print(f"Global Pixel Std Dev: {img_stats['pixel_std']:.4f}")

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 28)

    # Metadata Feature: Depth (z)
    # We analyze the relationship between Depth (z) and Salt Coverage

    # Correlation
    corr_depth_coverage = df["z"].corr(df["salt_coverage"])
    print(f"Correlation (Depth vs Salt Coverage): {corr_depth_coverage:.4f}")

    # Relationship interpretation
    if abs(corr_depth_coverage) < 0.1:
        interp = "Negligible"
    elif abs(corr_depth_coverage) < 0.3:
        interp = "Weak"
    elif abs(corr_depth_coverage) < 0.5:
        interp = "Moderate"
    else:
        interp = "Strong"
    direction = "Positive" if corr_depth_coverage > 0 else "Negative"
    print(f"  -> Indicates a {interp} {direction} relationship.")

    # Check if depth correlates with whether salt is present at all (Binary)
    df["has_salt"] = (df["salt_coverage"] > 0).astype(int)
    corr_depth_presence = df["z"].corr(df["has_salt"])
    print(f"Correlation (Depth vs Salt Presence Binary): {corr_depth_presence:.4f}")

    # Check if depth correlates with image intensity (if we had per-image intensity)
    # We can quickly compute per-image means to check this meta-feature relationship
    # This is "Unstructured (Meta-Feature) Relationships"

    # We'll re-loop quickly or just grab a sample if it was slow, but 2400 is fast.
    # Let's do a quick pass for mean intensity per image to correlate with depth.
    image_means = []
    for idx, row in df.iterrows():
        img_path = os.path.join(INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_means.append(np.mean(img))
        else:
            image_means.append(np.nan)

    df["image_mean_intensity"] = image_means
    corr_depth_intensity = df["z"].corr(df["image_mean_intensity"])

    print(f"Correlation (Depth vs Image Intensity): {corr_depth_intensity:.4f}")
    print(
        f"  -> Do deeper images tend to be darker/lighter? {corr_depth_intensity:.4f}"
    )


if __name__ == "__main__":
    main()
