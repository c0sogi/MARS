import os
import pandas as pd
import numpy as np
import cv2
import random
import sys

# Set constants
METADATA_FILE = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Distribution
    counts = df["label"].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")

    # Class balance ratios
    for label, count in counts.items():
        ratio = count / total
        label_name = "Dog" if label == 1 else "Cat"
        print(f"Class {label} ({label_name}): {count} samples ({ratio:.4f})")

    # Imbalance check
    min_class_ratio = counts.min() / total
    print(f"Minority Class Ratio: {min_class_ratio:.4f}")
    if min_class_ratio < 0.4:  # Arbitrary threshold for 'imbalance'
        print("Note: Dataset appears imbalanced.")
    else:
        print("Note: Dataset appears balanced.")
    print("")


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Accumulators for global pixel stats (R, G, B)
    # Storing sum and sum_squares to calculate mean/std without loading all images to RAM
    # OpenCV loads in BGR format
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixel_count = 0

    # Meta-features for relationship analysis
    meta_features = []

    # Iterate through dataset
    # We suppress errors for individual corrupt files but log them if necessary
    # Given the prompt, we assume standard dataset quality but handle exceptions gracefully

    valid_indices = []

    for idx, row in df.iterrows():
        filepath = os.path.join(INPUT_DIR, row["filepath"])

        # Read image
        img = cv2.imread(filepath)

        if img is None:
            continue

        h, w, c = img.shape

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels_list.append(c)

        # Pixel stats accumulation
        # Normalize to 0-1 range for calculation to avoid huge numbers, or keep 0-255
        # Standard practice is often reporting stats on 0-255 or 0-1. We will do 0-255.

        # Flatten spatial dimensions
        pixels = img.reshape(-1, 3)
        n_pixels = pixels.shape[0]

        # Accumulate
        # Note: img is BGR. We usually report RGB. Let's flip to RGB for stats.
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) -> This allocates memory.
        # Alternatively, just map indices: B=0, G=1, R=2 -> R=2, G=1, B=0

        # Sum per channel
        ch_sum = pixels.sum(axis=0)  # B, G, R sums
        ch_sq_sum = (pixels.astype(np.float64) ** 2).sum(axis=0)

        # Map BGR to RGB for global accumulation
        # channel_sum is [R_sum, G_sum, B_sum]
        channel_sum[0] += ch_sum[2]  # R
        channel_sum[1] += ch_sum[1]  # G
        channel_sum[2] += ch_sum[0]  # B

        channel_sq_sum[0] += ch_sq_sum[2]  # R
        channel_sq_sum[1] += ch_sq_sum[1]  # G
        channel_sq_sum[2] += ch_sq_sum[0]  # B

        total_pixel_count += n_pixels

        # Store meta features for later
        meta_features.append(
            {"width": w, "height": h, "aspect_ratio": w / h, "label": row["label"]}
        )
        valid_indices.append(idx)

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels_arr = np.array(channels_list)

    # 1. Dimensions
    print("Dimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # 2. Channels
    unique_channels, counts_channels = np.unique(channels_arr, return_counts=True)
    print("\nChannels:")
    for c, count in zip(unique_channels, counts_channels):
        c_type = "RGB" if c == 3 else ("Grayscale" if c == 1 else "Other")
        print(
            f"  {c} channels ({c_type}): {count} images ({count/len(channels_arr):.4f})"
        )

    # 3. Pixel Stats
    # Calculate global mean and std
    global_mean = channel_sum / total_pixel_count
    # Var = E[X^2] - (E[X])^2
    global_var = (channel_sq_sum / total_pixel_count) - (global_mean**2)
    global_std = np.sqrt(global_var)

    print("\nPixel Statistics (RGB, 0-255 scale):")
    print(
        f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    print(
        f"  Std:  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
    )

    # Return meta features dataframe for next section
    return pd.DataFrame(meta_features)


def analyze_relationships(meta_df):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)
    print("Unstructured (Meta-Feature) Relationships:")

    # We analyze if image dimensions correlate with the class (Dog vs Cat)
    # Point-Biserial correlation is essentially Pearson correlation when one variable is binary

    correlations = {}
    features = ["width", "height", "aspect_ratio"]

    for feat in features:
        corr = meta_df[feat].corr(meta_df["label"])
        correlations[feat] = corr

    print("Correlation with Target (Label=1 is Dog):")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # Compare Means per class
    print("\nMean Meta-features by Class:")
    means = meta_df.groupby("label")[features].mean()
    for label in [0, 1]:
        label_name = "Dog" if label == 1 else "Cat"
        print(f"  Class {label} ({label_name}):")
        for feat in features:
            print(f"    {feat}: {means.loc[label, feat]:.4f}")

    # Check for redundancy (collinearity between meta-features)
    print("\nMeta-feature Redundancy (Correlation > 0.90):")
    corr_matrix = meta_df[features].corr().abs()
    # Select upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if not high_corr:
        print("  No highly collinear meta-features found.")
    else:
        for col in high_corr:
            # Find row index
            rows = upper.index[upper[col] > 0.90].tolist()
            for row in rows:
                print(f"  {row} - {col}: {upper.loc[row, col]:.4f}")


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Image Analysis
    meta_df = analyze_images(df)

    # 3. Feature/Signal Relationships
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
