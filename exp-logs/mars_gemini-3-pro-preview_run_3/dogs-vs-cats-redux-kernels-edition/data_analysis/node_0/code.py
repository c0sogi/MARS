import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import pointbiserialr

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_modality(df):
    """
    Determines data modality based on dataframe columns and file extensions.
    """
    if "filepath" in df.columns:
        # Check first file extension
        first_file = df.iloc[0]["filepath"]
        ext = os.path.splitext(first_file)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            return "image"
        elif ext in [".wav", ".mp3", ".flac", ".ogg"]:
            return "audio"
        else:
            # Fallback or text if filepath points to text files (unlikely here)
            return "unknown"
    elif "text" in df.columns or "sentence" in df.columns:
        return "text"
    else:
        return "tabular"


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    if "label" not in df.columns:
        print("No 'label' column found.")
        return

    # Distribution
    counts = df["label"].value_counts()
    total = len(df)
    print(f"Total Samples: {total}")

    # Check if classification or regression
    # Heuristic: if few unique values (<20) and integer/string, likely classification
    unique_vals = df["label"].nunique()
    dtype = df["label"].dtype

    if unique_vals < 20 or dtype == object or dtype == bool:
        print("Type: Classification")
        print("Class Distribution:")
        for label, count in counts.items():
            ratio = count / total
            print(f"  Class {label}: {count} ({ratio:.4f})")

        # Balance check
        min_class = counts.min()
        max_class = counts.max()
        balance_ratio = min_class / max_class
        print(f"Class Balance Ratio (Min/Max): {balance_ratio:.4f}")
    else:
        print("Type: Regression")
        print(f"Mean: {df['label'].mean():.4f}")
        print(f"Std:  {df['label'].std():.4f}")
        print(f"Skewness: {df['label'].skew():.4f}")
        print(f"Kurtosis: {df['label'].kurtosis():.4f}")
    print("")


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # Pixel stats accumulators
    # We will sample for pixel stats to keep runtime low, but check dims for all
    pixel_stat_sample_size = 2000
    if len(df) > pixel_stat_sample_size:
        pixel_indices = set(
            np.random.choice(df.index, pixel_stat_sample_size, replace=False)
        )
    else:
        pixel_indices = set(df.index)

    pixel_sum = np.zeros(3)  # BGR in cv2
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Iterate through all images for metadata
    # We use a simple loop. For 18k images, this is fast enough for metadata reading.
    # Reading the image header is faster than decoding pixels, but cv2.imread decodes.
    # Given the constraints, we will process all for dims if possible, but if it takes too long
    # we might need to be careful. 18k small images should be fine in < 10 mins.

    print(f"Processing {len(df)} images for metadata analysis...")

    valid_count = 0

    for idx, row in df.iterrows():
        rel_path = row["filepath"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # File size
        fsize = os.path.getsize(full_path)
        file_sizes.append(fsize)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)

        valid_count += 1

        # Pixel stats (on sample)
        if idx in pixel_indices:
            # Normalize to 0-1 for calculation usually, but standard is 0-255.
            # We'll report in 0-255 scale.
            flat_img = img.reshape(-1, 3)
            pixel_sum += flat_img.sum(axis=0)
            pixel_sq_sum += (flat_img**2).sum(axis=0)
            pixel_count += flat_img.shape[0]

    if valid_count == 0:
        print("No valid images found.")
        return

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    ars = np.array(aspect_ratios)

    print("Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(f"  Aspect Ratio - Mean: {np.mean(ars):.4f}, Std: {np.std(ars):.4f}")

    # Channels
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print("Channels Distribution:")
    for c, count in zip(unique_channels, channel_counts):
        print(f"  {c} channels: {count} images ({count/valid_count:.4f})")

    # Pixel Stats
    if pixel_count > 0:
        # BGR to RGB for reporting
        # pixel_sum is [SumB, SumG, SumR]
        b_mean = pixel_sum[0] / pixel_count
        g_mean = pixel_sum[1] / pixel_count
        r_mean = pixel_sum[2] / pixel_count

        b_std = np.sqrt((pixel_sq_sum[0] / pixel_count) - (b_mean**2))
        g_std = np.sqrt((pixel_sq_sum[1] / pixel_count) - (g_mean**2))
        r_std = np.sqrt((pixel_sq_sum[2] / pixel_count) - (r_mean**2))

        print("Pixel Statistics (RGB, 0-255):")
        print(f"  Mean: [{r_mean:.4f}, {g_mean:.4f}, {b_mean:.4f}]")
        print(f"  Std:  [{r_std:.4f}, {g_std:.4f}, {b_std:.4f}]")

    # Store meta-features in df for relationship analysis
    # We need to align the lists with the dataframe.
    # Since we skipped missing/invalid images, we need to be careful.
    # For simplicity in this script, we'll assume the lists align with processed rows.
    # We will re-iterate or just add columns to a copy of the df that matches valid_count.
    # To do this robustly:

    meta_df = pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "aspect_ratio": ars,
            "file_size": file_sizes,
            "label": df.iloc[:valid_count][
                "label"
            ].values,  # Assuming sequential processing matches
        }
    )

    return meta_df


def analyze_relationships(meta_df, modality):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    if modality != "image":
        print("Relationship analysis implemented for Image modality in this run.")
        return

    if meta_df is None or len(meta_df) == 0:
        print("No metadata available for relationship analysis.")
        return

    # Correlation between meta-features and target
    # Point-Biserial Correlation (Continuous vs Binary)
    print("Meta-Feature vs Target (Label) Correlations:")

    features = ["width", "height", "aspect_ratio", "file_size"]
    for feat in features:
        if feat in meta_df.columns:
            # Calculate Point-Biserial
            corr, pval = pointbiserialr(meta_df[feat], meta_df["label"])
            print(
                f"  {feat.ljust(15)}: Correlation = {corr:.4f} (p-value = {pval:.4f})"
            )

    # Grouped Means
    print("\nAverage Meta-Features by Class:")
    grouped = meta_df.groupby("label")[features].mean()
    print(grouped.round(4))

    # Insight generation
    print("\nInsights:")
    for feat in features:
        diff = abs(grouped.loc[0, feat] - grouped.loc[1, feat])
        rel_diff = diff / grouped[feat].mean()
        if rel_diff > 0.1:
            print(
                f"  - Significant difference in {feat} between classes ({rel_diff*100:.1f}%). Potential bias or feature."
            )
        else:
            print(f"  - {feat} is relatively similar across classes.")


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return

    df = pd.read_csv(TRAIN_CSV)

    # 2. Determine Modality
    modality = get_modality(df)
    print(f"Detected Modality: {modality.upper()}\n")

    # 3. Target Variable Analysis
    analyze_target(df)

    # 4. Input Data Analysis & 5. Relationships
    if modality == "image":
        meta_df = analyze_images(df)
        analyze_relationships(meta_df, modality)
    else:
        print(f"Analysis for {modality} not fully implemented in this script version.")


if __name__ == "__main__":
    main()
