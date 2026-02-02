import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from scipy import stats

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    target_col = "category_id"

    # Distribution
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Imbalance
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Class Balance:")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Mean samples per class: {mean_samples:.4f}")
    print(f"  Median samples per class: {median_samples:.4f}")

    # Rare classes (< 1% frequency)
    # 1% of total samples
    threshold_1pct = total_samples * 0.01
    rare_classes = class_counts[class_counts < threshold_1pct]
    num_rare = len(rare_classes)
    pct_rare = (num_rare / num_classes) * 100

    print(f"Rare Classes (< 1% frequency): {num_rare} ({pct_rare:.2f}% of classes)")

    # Singleton classes
    singletons = class_counts[class_counts == 1]
    print(f"Singleton Classes (1 sample): {len(singletons)}")

    return class_counts


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Sample data
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Analyzing a subset of {len(sample_df)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []
    file_sizes = []

    # For pixel stats (Welford's algorithm or simple running sum)
    # Using running sum for simplicity and speed
    # Accumulators for R, G, B
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixels = 0

    missing_files = 0

    for _, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            # Check file size
            f_size = os.path.getsize(full_path)
            file_sizes.append(f_size)

            # Read image
            img = cv2.imread(full_path)

            if img is None:
                missing_files += 1
                continue

            # Dimensions (H, W, C)
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channel_counts.append(c)

            # Pixel stats
            # Convert to RGB if loaded as BGR by OpenCV
            if c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Flatten and accumulate
            # Normalize to 0-1 for stats calculation usually, but raw 0-255 is also fine.
            # Let's do 0-255 and normalize at print time if needed.
            img_flat = img.reshape(-1, c)

            # If grayscale, treat as 1 channel, but for sum array we might need handling
            if c == 1:
                # If grayscale, we can track just one channel or replicate.
                # Usually standard is to report 1 channel stats.
                # For simplicity, let's just track the first channel slot
                channel_sum[0] += img_flat[:, 0].sum()
                channel_sq_sum[0] += (img_flat[:, 0] ** 2).sum()
                total_pixels += w * h
            elif c == 3:
                channel_sum += img_flat.sum(axis=0)
                channel_sq_sum += (img_flat**2).sum(axis=0)
                total_pixels += w * h

        except Exception:
            missing_files += 1

    if missing_files > 0:
        print(f"Warning: {missing_files} images could not be read.")

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Dimensions
    print(f"Image Dimensions:")
    print(
        f"  Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"  Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"  Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}, Min={aspect_ratios.min():.4f}, Max={aspect_ratios.max():.4f}"
    )

    # Channels
    n_rgb = channel_counts.count(3)
    n_gray = channel_counts.count(1)
    print(f"Channels:")
    print(f"  RGB Images: {n_rgb} ({n_rgb/len(channel_counts)*100:.2f}%)")
    print(f"  Grayscale Images: {n_gray} ({n_gray/len(channel_counts)*100:.2f}%)")

    # Pixel Stats
    # Calculate global mean and std
    # Note: If mixed RGB and Grayscale, this stat is dominated by RGB structure.
    # Assuming mostly RGB based on typical datasets.
    if total_pixels > 0:
        pixel_means = channel_sum / total_pixels
        # std = sqrt(E[x^2] - (E[x])^2)
        pixel_stds = np.sqrt((channel_sq_sum / total_pixels) - (pixel_means**2))

        # Normalize to 0-1 range for reporting
        pixel_means_norm = pixel_means / 255.0
        pixel_stds_norm = pixel_stds / 255.0

        print(f"Pixel Statistics (Normalized 0-1):")
        print(
            f"  Mean (R, G, B): [{pixel_means_norm[0]:.4f}, {pixel_means_norm[1]:.4f}, {pixel_means_norm[2]:.4f}]"
        )
        print(
            f"  Std  (R, G, B): [{pixel_stds_norm[0]:.4f}, {pixel_stds_norm[1]:.4f}, {pixel_stds_norm[2]:.4f}]"
        )

    # Return metadata for relationship analysis
    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["aspect_ratio"] = aspect_ratios
    sample_df["file_size"] = file_sizes

    return sample_df


def analyze_relationships(sample_df, class_counts):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # 1. Meta-Feature vs Target Property (Class Frequency)
    # Do rare classes have different image characteristics (e.g., file size)?

    # Map class frequency to the sample dataframe
    sample_df["class_freq"] = sample_df["category_id"].map(class_counts)

    # Correlation between File Size and Class Frequency
    corr_size_freq, _ = stats.pearsonr(sample_df["file_size"], sample_df["class_freq"])
    print(f"Correlation (File Size vs Class Frequency): {corr_size_freq:.4f}")

    # Correlation between Aspect Ratio and Class Frequency
    corr_ar_freq, _ = stats.pearsonr(sample_df["aspect_ratio"], sample_df["class_freq"])
    print(f"Correlation (Aspect Ratio vs Class Frequency): {corr_ar_freq:.4f}")

    # 2. Meta-Feature vs Target Class (ANOVA)
    # Do different classes have significantly different Aspect Ratios?
    # We take the top 5 most frequent classes in the sample to perform ANOVA
    top_classes = sample_df["category_id"].value_counts().head(5).index.tolist()

    groups = []
    for cls in top_classes:
        groups.append(sample_df[sample_df["category_id"] == cls]["aspect_ratio"].values)

    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        print(
            f"ANOVA (Aspect Ratio across Top 5 Classes): F-stat={f_stat:.4f}, p-value={p_val:.4f}"
        )
        if p_val < 0.05:
            print("  -> Significant difference in Aspect Ratios between top classes.")
        else:
            print(
                "  -> No significant difference in Aspect Ratios between top classes."
            )

    # 3. Check for Outliers in Meta-Features
    # IQR method for Aspect Ratio
    Q1 = sample_df["aspect_ratio"].quantile(0.25)
    Q3 = sample_df["aspect_ratio"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (
        (sample_df["aspect_ratio"] < (Q1 - 1.5 * IQR))
        | (sample_df["aspect_ratio"] > (Q3 + 1.5 * IQR))
    ).sum()
    print(
        f"Aspect Ratio Outliers (IQR Method): {outliers} ({outliers/len(sample_df)*100:.2f}%)"
    )


def main():
    # Ensure reproducibility
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Data
    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    class_counts = analyze_target(df)

    # 2. Image Analysis (on subset)
    sample_df_with_meta = analyze_images(df)

    # 3. Relationships
    analyze_relationships(sample_df_with_meta, class_counts)


if __name__ == "__main__":
    main()
