import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
SEED = 42
SAMPLE_SIZE = 5000  # Number of images to sample for pixel-level analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df):
    print("--- TARGET VARIABLE ANALYSIS ---")

    target_col = "label"
    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")
    print("Class Distribution:")
    for label, count in counts.items():
        ratio = count / total
        label_name = "Stego (1)" if label == 1 else "Cover (0)"
        print(f"  {label_name}: {count} ({ratio:.4f})")

    # Check imbalance
    if len(counts) > 1:
        ratio_balance = counts.min() / counts.max()
        print(f"Minority/Majority Ratio: {ratio_balance:.4f}")
    else:
        print("Only one class detected.")
    print("")


def analyze_images(df):
    print("--- INPUT DATA ANALYSIS (IMAGE) ---")

    # Sampling for efficiency
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(
        f"Analysis performed on a stratified random sample of {len(sample_df)} images."
    )

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # Running stats for global mean/std
    # Using Welford's online algorithm or simple accumulation for approximation
    # Given the constraints, simple accumulation of sum and sum_sq is sufficient for global stats
    total_pixel_sum = np.zeros(3)  # Assuming RGB max
    total_pixel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    # Meta-features for relationship analysis later
    meta_features = []

    for _, row in sample_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # File Size
        try:
            f_size = os.path.getsize(file_path)
        except OSError:
            continue

        file_sizes.append(f_size)

        # Read Image
        img = cv2.imread(file_path)
        if img is None:
            continue

        # OpenCV loads as BGR, convert to RGB for standard reporting
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)

        # Pixel Stats (Per image)
        # Normalize to 0-1 for calculation
        img_norm = img / 255.0
        img_pixels = img_norm.reshape(-1, 3)

        n_pixels = img_pixels.shape[0]
        total_pixel_count += n_pixels
        total_pixel_sum += img_pixels.sum(axis=0)
        total_pixel_sq_sum += (img_pixels**2).sum(axis=0)

        # Store meta-features for this row
        meta_features.append(
            {
                "label": row["label"],
                "file_size_bytes": f_size,
                "width": w,
                "height": h,
                "mean_intensity": img_norm.mean(),
            }
        )

    # 1. Dimensions
    w_series = pd.Series(widths)
    h_series = pd.Series(heights)
    ar_series = pd.Series(aspect_ratios)

    print("Dimensions:")
    print(
        f"  Width  - Mean: {w_series.mean():.4f}, Std: {w_series.std():.4f}, Min: {w_series.min()}, Max: {w_series.max()}"
    )
    print(
        f"  Height - Mean: {h_series.mean():.4f}, Std: {h_series.std():.4f}, Min: {h_series.min()}, Max: {h_series.max()}"
    )
    print(f"  Aspect Ratio - Mean: {ar_series.mean():.4f}, Std: {ar_series.std():.4f}")

    # 2. Channels
    c_series = pd.Series(channels)
    print("Channels:")
    print(f"  Distribution: {c_series.value_counts().to_dict()}")

    # 3. Pixel Stats
    if total_pixel_count > 0:
        global_mean = total_pixel_sum / total_pixel_count
        # E[X^2] - (E[X])^2
        global_var = (total_pixel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print("Pixel Statistics (Normalized 0-1, RGB):")
        print(
            f"  Global Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"  Global Std:  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
        )

    print("")
    return pd.DataFrame(meta_features)


def analyze_relationships(meta_df):
    print("--- FEATURE/SIGNAL RELATIONSHIPS ---")

    if meta_df.empty:
        print("No meta-features extracted.")
        return

    # Unstructured (Meta-Feature) Relationships
    # Correlate metadata with target

    print("Meta-Feature Correlations with Target (Label):")

    # Numerical features to check
    features = ["file_size_bytes", "width", "height", "mean_intensity"]

    # Filter features that actually vary
    valid_features = [f for f in features if meta_df[f].std() > 0]

    correlations = meta_df[valid_features + ["label"]].corr()["label"].drop("label")

    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    print("\nMeta-Feature Analysis by Class:")
    groupby_stats = meta_df.groupby("label")[valid_features].mean()

    for feature in valid_features:
        val_0 = groupby_stats.loc[0, feature] if 0 in groupby_stats.index else 0
        val_1 = groupby_stats.loc[1, feature] if 1 in groupby_stats.index else 0
        diff = val_1 - val_0
        print(
            f"  {feature} (Avg): Cover={val_0:.4f}, Stego={val_1:.4f}, Diff={diff:.4f}"
        )

    print("")


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return

    df = pd.read_csv(TRAIN_CSV)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Image Analysis
    meta_df = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
