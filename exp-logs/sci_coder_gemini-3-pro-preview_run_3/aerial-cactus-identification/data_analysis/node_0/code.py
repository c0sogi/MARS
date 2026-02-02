import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import pointbiserialr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df, target_col):
    print("==== TARGET VARIABLE ANALYSIS ====")
    counts = df[target_col].value_counts()
    ratios = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {len(df)}")
    print("Class Distribution:")
    for cls, count in counts.items():
        ratio = ratios[cls]
        print(f"  Class {cls}: {count} ({ratio:.4f})")

    # Check for imbalance
    majority_class_ratio = ratios.max()
    if majority_class_ratio > 0.6:
        print(
            f"Imbalance Detected: Majority class constitutes {majority_class_ratio:.4f} of the data."
        )
    else:
        print("Class Balance: The dataset is relatively balanced.")


def analyze_images(df, input_dir):
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Accumulators for global stats
    channel_sums = np.zeros(3)
    channel_sq_sums = np.zeros(3)
    total_pixels = 0

    # Accumulators for meta-features
    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Meta-features for relationship analysis
    # We will store: [brightness_r, brightness_g, brightness_b, contrast, target]
    meta_features = []

    # Iterate through images
    # Using a list comprehension or loop. Since dataset is small (32x32 images), loop is fast.
    valid_count = 0

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)
        target = row["has_cactus"]

        # Read image
        img = cv2.imread(full_path)

        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        # Update dimension stats
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels_list.append(c)

        # Update global pixel stats
        # Normalize to 0-1 for calculation to avoid overflow, or keep as 0-255?
        # Requirement usually implies 0-255 or standardized. Let's calculate on 0-255 scale.
        img_flat = img.reshape(-1, 3)
        channel_sums += img_flat.sum(axis=0)
        channel_sq_sums += (img_flat**2).sum(axis=0)
        total_pixels += h * w

        # Calculate per-image meta-features for relationship analysis
        # Brightness: Mean pixel value
        mean_intensity = img.mean()
        # Contrast: Std dev of pixel values
        std_intensity = img.std()

        meta_features.append(
            {"brightness": mean_intensity, "contrast": std_intensity, "target": target}
        )

        valid_count += 1

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
    unique_channels = np.unique(channels_list)
    print(f"Channels Distribution: {unique_channels}")
    if len(unique_channels) == 1 and unique_channels[0] == 3:
        print("  All images are RGB.")
    elif len(unique_channels) == 1 and unique_channels[0] == 1:
        print("  All images are Grayscale.")
    else:
        print("  Mixed channel counts detected.")

    # 3. Pixel Stats (Global)
    # Mean = Sum / N
    # Std = Sqrt( (SumSq / N) - Mean^2 )
    global_means = channel_sums / total_pixels
    global_stds = np.sqrt((channel_sq_sums / total_pixels) - (global_means**2))

    print("Pixel Statistics (Global, RGB, 0-255 scale):")
    print(f"  Red Channel:   Mean={global_means[0]:.4f}, Std={global_stds[0]:.4f}")
    print(f"  Green Channel: Mean={global_means[1]:.4f}, Std={global_stds[1]:.4f}")
    print(f"  Blue Channel:  Mean={global_means[2]:.4f}, Std={global_stds[2]:.4f}")

    return pd.DataFrame(meta_features)


def analyze_relationships(meta_df):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")
    print(
        "Analyzing relationship between image meta-features and target (has_cactus)..."
    )

    features = ["brightness", "contrast"]

    for feat in features:
        # Calculate correlation
        # Point-biserial correlation is used when one variable is continuous and the other is binary
        corr, p_val = pointbiserialr(meta_df[feat], meta_df["target"])

        # Calculate means per class
        mean_pos = meta_df[meta_df["target"] == 1][feat].mean()
        mean_neg = meta_df[meta_df["target"] == 0][feat].mean()

        print(f"Feature: {feat.capitalize()}")
        print(f"  Correlation with Target: {corr:.4f} (p-value: {p_val:.4f})")
        print(f"  Mean (Cactus=1): {mean_pos:.4f}")
        print(f"  Mean (Cactus=0): {mean_neg:.4f}")
        print(f"  Difference: {mean_pos - mean_neg:.4f}")

    print("\nInterpretation:")
    print(
        "  Higher positive correlation implies the feature value increases when a cactus is present."
    )
    print(
        "  Negative correlation implies the feature value decreases when a cactus is present."
    )


def main():
    set_seed()

    # Paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_metadata.csv"

    # Load Data
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    df_train = pd.read_csv(METADATA_PATH)

    # 1. Target Variable Analysis
    analyze_target(df_train, "has_cactus")

    # 2. Input Data Analysis (Image)
    # This function returns a dataframe of meta-features extracted during the scan
    meta_df = analyze_images(df_train, INPUT_DIR)

    # 3. Feature/Signal Relationships
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
