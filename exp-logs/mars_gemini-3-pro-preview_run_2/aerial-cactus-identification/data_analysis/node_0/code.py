import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from scipy.stats import pointbiserialr

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42


# --- Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


def main():
    # 1. Data Integrity & Loading
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Ensure full paths
    # Metadata contains relative paths like 'train/id.jpg'
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS         ")
    print("========================================")

    # 2. Target Variable Analysis
    analyze_target(df)

    # 3. Input Data Analysis (Image Modality)
    # We extract meta-features here to be used in section 4 as well
    meta_features_df = analyze_images(df)

    # 4. Feature/Signal Relationships
    analyze_relationships(meta_features_df, df["has_cactus"])


def analyze_target(df):
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "has_cactus"
    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: '{target_col}'")
    print(f"Total Samples: {len(df)}")

    # Distribution
    print("\nDistribution:")
    for label, count in counts.items():
        prop = proportions[label]
        print(f"Class {label}: {count} samples ({prop * 100:.4f}%)")

    # Imbalance
    if len(counts) == 2:
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        ratio = counts[majority_class] / counts[minority_class]
        print(f"\nClass Imbalance Ratio (Majority/Minority): {ratio:.4f}")
    else:
        print("\nClass Imbalance: Multi-class or Single-class detected.")


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("-" * 30)

    # We will collect stats to avoid multiple passes
    widths = []
    heights = []
    channels = []

    # Pixel stats accumulators
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    # Meta-features for relationship analysis
    meta_brightness = []
    meta_contrast = []
    meta_red_mean = []
    meta_green_mean = []
    meta_blue_mean = []

    # Process images
    # Using a loop since dataset is small (11k 32x32 images)
    # For very large datasets, we would sample.

    valid_images_count = 0

    for idx, row in df.iterrows():
        fpath = row["full_path"]

        # cv2.imread loads as BGR
        img = cv2.imread(fpath)

        if img is None:
            continue

        valid_images_count += 1
        h, w, c = img.shape

        widths.append(w)
        heights.append(h)
        channels.append(c)

        # Convert to RGB for correct channel stats
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Accumulate global stats
        # Normalize to 0-1 for calculation stability if needed,
        # but prompt implies reporting pixel values. We'll report 0-255 stats.
        pixels = img_rgb.flatten()
        pixel_sum += np.sum(pixels)
        pixel_sq_sum += np.sum(pixels**2)
        total_pixels += len(pixels)

        # Meta-features extraction
        # Mean intensity (Brightness)
        mean_intensity = np.mean(img_rgb)
        meta_brightness.append(mean_intensity)

        # Standard Deviation (Contrast)
        std_intensity = np.std(img_rgb)
        meta_contrast.append(std_intensity)

        # Channel Means
        meta_red_mean.append(np.mean(img_rgb[:, :, 0]))
        meta_green_mean.append(np.mean(img_rgb[:, :, 1]))
        meta_blue_mean.append(np.mean(img_rgb[:, :, 2]))

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = widths / heights

    print("Dimensions:")
    print(
        f"Image Widths:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Image Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels Analysis
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print("\nChannels:")
    for c, count in zip(unique_channels, channel_counts):
        c_type = "RGB" if c == 3 else ("Grayscale" if c == 1 else "Unknown")
        print(f"{c} Channels ({c_type}): {count} images")

    # Pixel Stats Analysis (Global)
    # E[X] = Sum(x) / N
    # Var(X) = E[X^2] - (E[X])^2
    global_mean = pixel_sum / total_pixels
    global_variance = (pixel_sq_sum / total_pixels) - (global_mean**2)
    global_std = np.sqrt(global_variance)

    print("\nPixel Statistics (Global 0-255):")
    print(f"Mean Pixel Value: {global_mean:.4f}")
    print(f"Std Dev Pixel Value: {global_std:.4f}")

    # Return DataFrame of meta-features for next section
    return pd.DataFrame(
        {
            "brightness": meta_brightness,
            "contrast": meta_contrast,
            "red_mean": meta_red_mean,
            "green_mean": meta_green_mean,
            "blue_mean": meta_blue_mean,
        }
    )


def analyze_relationships(meta_df, target):
    print("\nFEATURE/SIGNAL RELATIONSHIPS (UNSTRUCTURED)")
    print("-" * 30)

    # Combine for analysis
    df = meta_df.copy()
    df["target"] = target.values

    print("Relationship between Image Meta-Features and Target (has_cactus):")

    features = ["brightness", "contrast", "red_mean", "green_mean", "blue_mean"]

    # Correlation Analysis
    print("\nPoint-Biserial Correlation with Target (1=Cactus, 0=No Cactus):")
    print("(Positive value indicates feature is higher in Cactus images)")

    correlations = []
    for feat in features:
        # Point Biserial is used for Continuous vs Binary
        corr, pval = pointbiserialr(df[feat], df["target"])
        correlations.append((feat, corr))
        print(f"{feat.ljust(12)}: Correlation = {corr:.4f} (p-value = {pval:.4f})")

    # Grouped Means
    print("\nAverage Feature Values by Class:")
    grouped = df.groupby("target")[features].mean()

    # Format the grouped dataframe for printing
    for feat in features:
        val_0 = grouped.loc[0, feat]
        val_1 = grouped.loc[1, feat]
        diff = val_1 - val_0
        print(
            f"{feat.ljust(12)}: No Cactus (0) = {val_0:.4f}, Cactus (1) = {val_1:.4f}, Diff = {diff:.4f}"
        )

    # Redundancy check among meta-features (Collinearity)
    print("\nMeta-Feature Redundancy (Correlation > 0.90):")
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [column for column in upper.columns if any(upper[column] > 0.90)]
    found_redundancy = False
    for col in high_corr_pairs:
        for row in upper.index:
            if upper.loc[row, col] > 0.90:
                print(
                    f"High Correlation ({upper.loc[row, col]:.4f}) between '{row}' and '{col}'"
                )
                found_redundancy = True

    if not found_redundancy:
        print("No highly collinear meta-feature pairs found.")


if __name__ == "__main__":
    main()
