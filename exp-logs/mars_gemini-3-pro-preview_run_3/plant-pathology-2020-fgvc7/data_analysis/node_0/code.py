import os
import pandas as pd
import numpy as np
import cv2
import random

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_targets(df):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Identify the target label from the 'stratify_label' column created in metadata
    # or derive it from the one-hot columns.
    target_col = "stratify_label"

    if target_col not in df.columns:
        # Fallback if stratify_label is missing, though metadata guarantees it
        label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
        df[target_col] = df[label_cols].idxmax(axis=1)

    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {len(df)}")
    print("\nClass Distribution:")
    for label, count in counts.items():
        prop = proportions[label]
        print(f"  - {label:<20}: {count} ({prop:.4f})")

    # Check for imbalance
    max_prop = proportions.max()
    min_prop = proportions.min()
    ratio = max_prop / min_prop
    print(f"\nImbalance Ratio (Max/Min class): {ratio:.4f}")
    if ratio > 5:
        print("  -> ALERT: Significant class imbalance detected.")
    elif ratio > 2:
        print("  -> NOTE: Moderate class imbalance detected.")
    else:
        print("  -> Class distribution is relatively balanced.")


def analyze_images(df):
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Accumulators for global pixel stats (using Welford's algorithm or simple sum for mean)
    # Given dataset size (~1300), we can sum values.
    # We will compute stats on 0-255 scale.

    total_pixels = 0
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)

    # Store meta-features for relationship analysis
    meta_features = {
        "image_mean_brightness": [],
        "image_contrast": [],  # std dev of intensity
        "image_area": [],
    }

    valid_indices = []

    # Iterate through images
    # Suppress potential warnings from libraries
    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            # Should not happen based on metadata checks
            continue

        img = cv2.imread(full_path)
        if img is None:
            continue

        # OpenCV loads as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels_list.append(c)

        num_pixels = w * h
        total_pixels += num_pixels

        # Pixel stats for this image
        # Normalize to 0-1 for internal calc if needed, but keeping 0-255 for reporting is standard for "pixel values"
        img_float = img.astype(np.float64)

        # Update global accumulators
        # Sum over height and width
        ch_sum = np.sum(img_float, axis=(0, 1))
        ch_sq_sum = np.sum(img_float**2, axis=(0, 1))

        channel_sum += ch_sum
        channel_sq_sum += ch_sq_sum

        # Meta features for this image (using simple grayscale intensity for brightness/contrast)
        # Luminance = 0.299 R + 0.587 G + 0.114 B
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        meta_features["image_mean_brightness"].append(np.mean(gray))
        meta_features["image_contrast"].append(np.std(gray))
        meta_features["image_area"].append(w * h)
        valid_indices.append(idx)

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels_list = np.array(channels_list)

    # 1. Dimensions
    print("Dimensions:")
    print(
        f"  - Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  - Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  - Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # 2. Channels
    unique_channels, counts_channels = np.unique(channels_list, return_counts=True)
    print("\nChannels:")
    for c, count in zip(unique_channels, counts_channels):
        print(f"  - {c} channels: {count} images")

    # 3. Pixel Stats (Global)
    # Global Mean = Total Sum / Total Pixels
    global_mean = channel_sum / total_pixels

    # Global Std = sqrt( E[x^2] - (E[x])^2 )
    global_sq_mean = channel_sq_sum / total_pixels
    global_std = np.sqrt(global_sq_mean - global_mean**2)

    print("\nGlobal Pixel Statistics (RGB, 0-255 scale):")
    print(
        f"  - Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    print(
        f"  - Std:  R={global_std[0]:.4f},  G={global_std[1]:.4f},  B={global_std[2]:.4f}"
    )

    # Return df with added meta features for relationship analysis
    # Filter df to valid indices just in case
    df_valid = df.loc[valid_indices].copy()
    for k, v in meta_features.items():
        df_valid[k] = v
    df_valid["aspect_ratio"] = aspect_ratios

    return df_valid


def analyze_relationships(df):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    target_col = "stratify_label"

    # Group by target and calculate mean of meta-features
    print(f"Meta-Feature Analysis by Target ({target_col}):")

    features_to_analyze = [
        "image_mean_brightness",
        "image_contrast",
        "aspect_ratio",
        "image_area",
    ]

    groupby_obj = df.groupby(target_col)[features_to_analyze]
    means = groupby_obj.mean()
    stds = groupby_obj.std()

    for feature in features_to_analyze:
        print(f"\nFeature: {feature}")
        print(f"{'Class':<20} | {'Mean':<10} | {'Std':<10}")
        print("-" * 46)
        for label in means.index:
            m = means.loc[label, feature]
            s = stds.loc[label, feature]
            print(f"{label:<20} | {m:.4f}     | {s:.4f}")

    # Check for correlation between brightness and disease type?
    # Since target is categorical, we look at the separation in means above.
    # We can briefly comment on potential outliers or specific patterns if obvious,
    # but the table serves as the primary report.


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # 1. Target Analysis
    analyze_targets(df)

    # 2. Image Analysis
    df_enriched = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(df_enriched)


if __name__ == "__main__":
    main()
