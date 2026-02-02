import os
import pandas as pd
import numpy as np
import cv2
import random
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    # 1. Setup and Data Integrity
    set_seed()

    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    # Load training metadata
    # This ensures we only look at the training set defined in the previous step
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # Construct full paths
    # The metadata contains relative paths like 'train_images/id.png'
    # We need to prepend the INPUT_DIR
    df_train["full_path"] = df_train["file_path"].apply(
        lambda x: os.path.join(INPUT_DIR, x)
    )

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    target_col = "diagnosis"

    # Distribution and Imbalance
    class_counts = df_train[target_col].value_counts().sort_index()
    total_samples = len(df_train)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total_samples}")
    print("Class Distribution:")

    for label, count in class_counts.items():
        ratio = count / total_samples
        print(f"  Class {label}: {count} samples ({ratio:.4f})")

    # Check for imbalance
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()
    imbalance_ratio = (
        max_class_count / min_class_count if min_class_count > 0 else float("inf")
    )

    print(f"Imbalance Ratio (Max/Min class): {imbalance_ratio:.4f}")
    if imbalance_ratio > 10:
        print("  NOTE: Severe class imbalance detected.")
    elif imbalance_ratio > 2:
        print("  NOTE: Moderate class imbalance detected.")
    else:
        print("  NOTE: Classes are relatively balanced.")

    # ==========================================
    # 3. Input Data Analysis (Image)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Initialize accumulators for global stats
    # We will compute mean/std using running sums to save memory
    # Stats calculated on [0, 1] scale
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    # Lists to store meta-features for relationship analysis
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []  # Global brightness per image
    channel_counts = {}

    # Iterate through images
    # We use a sample if the dataset is massive, but 2636 is small enough to process all
    # to get accurate stats.

    valid_indices = []

    for idx, row in df_train.iterrows():
        fpath = row["full_path"]

        # Read image
        try:
            # cv2 reads in BGR
            img = cv2.imread(fpath)
            if img is None:
                continue

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape

            # Store dimensions
            widths.append(w)
            heights.append(h)
            ar = w / h
            aspect_ratios.append(ar)

            # Track channel counts
            channel_counts[c] = channel_counts.get(c, 0) + 1

            # Update global pixel stats
            # Normalize to [0, 1] for calculation
            img_norm = img / 255.0

            # Sum of pixels for this image
            img_sum = img_norm.sum(axis=(0, 1))
            img_sq_sum = (img_norm**2).sum(axis=(0, 1))

            channel_sum += img_sum
            channel_sq_sum += img_sq_sum
            total_pixel_count += h * w

            # Store mean intensity for this image (average of R,G,B means)
            # This is a meta-feature for correlation analysis
            mean_intensities.append(img_norm.mean())
            valid_indices.append(idx)

        except Exception as e:
            # In case of corrupt file, skip
            continue

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    mean_intensities = np.array(mean_intensities)

    # Calculate Global Pixel Stats
    if total_pixel_count > 0:
        global_mean = channel_sum / total_pixel_count
        # Var = E[x^2] - (E[x])^2
        global_var = (channel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)
    else:
        global_mean = np.zeros(3)
        global_std = np.zeros(3)

    # Report Dimensions
    print("Image Dimensions:")
    print(
        f"  Widths:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"  Heights: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"  Aspect Ratios: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}, Min={aspect_ratios.min():.4f}, Max={aspect_ratios.max():.4f}"
    )

    # Report Channels
    print("Channel Distribution:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images")

    # Report Pixel Stats
    print("Global Pixel Statistics (RGB, normalized [0, 1]):")
    print(
        f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    print(
        f"  Std:  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
    )

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Create a DataFrame for meta-features
    # Filter original df to only include valid images processed
    df_meta = df_train.loc[valid_indices].copy()
    df_meta["width"] = widths
    df_meta["height"] = heights
    df_meta["aspect_ratio"] = aspect_ratios
    df_meta["mean_intensity"] = mean_intensities

    # Group by Target
    print("Meta-Feature Means by Target Class:")
    grouped = df_meta.groupby(target_col)[
        ["width", "height", "aspect_ratio", "mean_intensity"]
    ].mean()
    print(grouped.applymap(lambda x: f"{x:.4f}"))

    # Correlations
    # Since target is ordinal (0-4), we can check Spearman correlation
    print("\nCorrelation with Target (Spearman):")
    correlations = df_meta[
        ["diagnosis", "width", "height", "aspect_ratio", "mean_intensity"]
    ].corr(method="spearman")["diagnosis"]

    for feat in ["width", "height", "aspect_ratio", "mean_intensity"]:
        corr_val = correlations.get(feat, 0)
        print(f"  {feat}: {corr_val:.4f}")

    # Check for strong redundancy between meta-features
    print("\nMeta-Feature Redundancy (Pearson > 0.90):")
    meta_corr = df_meta[["width", "height", "aspect_ratio", "mean_intensity"]].corr(
        method="pearson"
    )
    found_redundancy = False
    for i in range(len(meta_corr.columns)):
        for j in range(i + 1, len(meta_corr.columns)):
            if abs(meta_corr.iloc[i, j]) > 0.90:
                print(
                    f"  {meta_corr.columns[i]} & {meta_corr.columns[j]}: {meta_corr.iloc[i, j]:.4f}"
                )
                found_redundancy = True

    if not found_redundancy:
        print("  No highly collinear meta-features found.")


if __name__ == "__main__":
    main()
