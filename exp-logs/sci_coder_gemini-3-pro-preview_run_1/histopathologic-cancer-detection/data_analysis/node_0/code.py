import os
import cv2
import numpy as np
import pandas as pd
import random
import time
from scipy import stats

# ==========================================
# Configuration & Setup
# ==========================================
SEED = 42
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 10000  # Number of images to sample for pixel-level analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_image(rel_path):
    """Loads an image from the input directory."""
    full_path = os.path.join(INPUT_DIR, rel_path)
    # cv2.imread might fail if path doesn't exist or file is corrupt
    if not os.path.exists(full_path):
        return None
    try:
        # Load as is (unchanged) to detect grayscale vs RGB
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        # If OpenCV fails to load (returns None), handle it
        if img is None:
            return None
        # Convert BGR to RGB if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def analyze_target(df):
    """Analyzes the target variable distribution."""
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    counts = df["label"].value_counts()
    proportions = df["label"].value_counts(normalize=True)

    print(f"Target Variable: 'label' (Binary Classification)")
    print(f"Total Samples: {len(df)}")
    print(f"Class 0 (No Tumor): {counts.get(0, 0)} ({proportions.get(0, 0):.4f})")
    print(f"Class 1 (Tumor):    {counts.get(1, 0)} ({proportions.get(1, 0):.4f})")

    ratio = counts.get(0, 0) / max(1, counts.get(1, 0))
    print(f"Class Balance Ratio (0:1): {ratio:.4f}")

    if ratio > 1.5 or ratio < 0.66:
        print("Observation: The dataset shows moderate class imbalance.")
    else:
        print("Observation: The dataset is relatively balanced.")
    print("")


def analyze_images(df):
    """Analyzes image dimensions, channels, and pixel stats via sampling."""
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Stratified Sampling
    if len(df) > SAMPLE_SIZE:
        # Group by label and sample
        sample_df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), int(SAMPLE_SIZE * len(x) / len(df))), random_state=SEED
            )
        )
    else:
        sample_df = df.copy()

    print(f"Analysis performed on a stratified sample of {len(sample_df)} images.")

    widths = []
    heights = []
    channels = []
    aspect_ratios = []

    # Accumulators for Welford's online algorithm or simple aggregation
    # We will use simple aggregation of means and variances to compute global stats
    # Global Mean = Mean(Image Means)
    # Global Var = Mean(Image Vars) + Var(Image Means)

    img_means = []
    img_vars = []

    # Meta-features for correlation analysis
    meta_brightness = []
    meta_contrast = []
    meta_labels = []

    start_time = time.time()

    for idx, row in sample_df.iterrows():
        img = load_image(row["file_path"])

        if img is None:
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        channels.append(c)
        aspect_ratios.append(w / h)

        # Normalize to 0-1 for stats calculation to avoid overflow
        img_norm = img.astype(np.float32) / 255.0

        # Calculate mean and var per channel for this image
        # If grayscale, reshape to (H, W, 1) to keep logic consistent
        if c == 1:
            img_norm = img_norm[:, :, np.newaxis]

        # Compute mean/var across spatial dimensions (axis 0 and 1)
        # Result shape: (C,)
        mu = np.mean(img_norm, axis=(0, 1))
        var = np.var(img_norm, axis=(0, 1))

        img_means.append(mu)
        img_vars.append(var)

        # For meta-feature analysis, take the average across channels
        meta_brightness.append(np.mean(mu))
        meta_contrast.append(
            np.sqrt(np.mean(var))
        )  # Approximate contrast as RMS contrast
        meta_labels.append(row["label"])

    # Convert lists to numpy arrays
    img_means = np.array(img_means)  # Shape: (N, C)
    img_vars = np.array(img_vars)  # Shape: (N, C)

    # 1. Dimensions
    print(f"Dimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}")

    # 2. Channels
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print(f"Channels Distribution:")
    for c, count in zip(unique_channels, channel_counts):
        print(f"  {c} channels: {count} images ({count/len(channels):.4f})")

    # 3. Pixel Stats (Global)
    # Assuming all images have same number of channels for the global stat calculation
    # If mixed (RGB and Gray), this simple aggregation fails. We check majority.
    majority_channel_count = unique_channels[np.argmax(channel_counts)]

    # Filter stats to only include majority channel count for valid global stats
    valid_indices = [i for i, c in enumerate(channels) if c == majority_channel_count]
    valid_means = img_means[valid_indices]
    valid_vars = img_vars[valid_indices]

    # Global Mean = Mean of Image Means
    global_mean = np.mean(valid_means, axis=0)

    # Global Variance = Mean(Image Vars) + Var(Image Means)
    global_var = np.mean(valid_vars, axis=0) + np.var(valid_means, axis=0)
    global_std = np.sqrt(global_var)

    print(
        f"Pixel Statistics (Normalized 0-1, calculated on {majority_channel_count}-channel images):"
    )
    print(f"  Global Mean per Channel: {global_mean}")
    print(f"  Global Std  per Channel: {global_std}")

    return pd.DataFrame(
        {"brightness": meta_brightness, "contrast": meta_contrast, "label": meta_labels}
    )


def analyze_relationships(meta_df):
    """Analyzes relationships between extracted meta-features and the target."""
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # 1. Correlation
    # Point-Biserial Correlation (Continuous Feature vs Binary Target)
    r_bright, p_bright = stats.pointbiserialr(meta_df["label"], meta_df["brightness"])
    r_contrast, p_contrast = stats.pointbiserialr(meta_df["label"], meta_df["contrast"])

    print("Meta-Feature Correlations with Target (Point-Biserial):")
    print(f"  Brightness vs Label: Correlation={r_bright:.4f} (p={p_bright:.4e})")
    print(f"  Contrast   vs Label: Correlation={r_contrast:.4f} (p={p_contrast:.4e})")

    # 2. Distribution Comparison
    print("\nMeta-Feature Distribution by Class:")

    for feature in ["brightness", "contrast"]:
        pos_vals = meta_df[meta_df["label"] == 1][feature]
        neg_vals = meta_df[meta_df["label"] == 0][feature]

        print(f"  {feature.capitalize()}:")
        print(
            f"    Class 0 (No Tumor): Mean={pos_vals.mean():.4f}, Std={pos_vals.std():.4f}"
        )
        print(
            f"    Class 1 (Tumor):    Mean={neg_vals.mean():.4f}, Std={neg_vals.std():.4f}"
        )

        # Simple Cohen's d effect size
        pooled_std = np.sqrt((pos_vals.std() ** 2 + neg_vals.std() ** 2) / 2)
        cohens_d = (pos_vals.mean() - neg_vals.mean()) / pooled_std
        print(f"    Effect Size (Cohen's d): {cohens_d:.4f}")


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Image Analysis
    meta_df = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
