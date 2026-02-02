import os
import numpy as np
import pandas as pd
import cv2
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def calculate_rle_area(rle):
    """Calculates the total number of pixels in an RLE mask."""
    if pd.isna(rle) or rle == "":
        return 0
    # RLE format: start length start length ...
    # We sum every second element (lengths)
    s = rle.split()
    return sum(int(x) for x in s[1::2])


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Target Variable Analysis
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    # Preprocessing RLE to get area and binary existence
    df["mask_area"] = df["segmentation"].apply(calculate_rle_area)
    df["has_mask"] = (df["mask_area"] > 0).astype(int)

    # Distribution of Classes
    class_counts = df["class"].value_counts()
    print("Class Distribution (Rows per class):")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Imbalance (Mask vs No Mask)
    total_samples = len(df)
    positive_samples = df["has_mask"].sum()
    negative_samples = total_samples - positive_samples
    pos_ratio = positive_samples / total_samples

    print(f"\nMask Existence Imbalance:")
    print(f"  Total Slices: {total_samples}")
    print(f"  Slices with Mask: {positive_samples} ({pos_ratio:.4f})")
    print(f"  Slices without Mask: {negative_samples} ({1 - pos_ratio:.4f})")

    # Breakdown by Class
    print("\nPositive Ratio by Class:")
    class_groups = df.groupby("class")["has_mask"].mean()
    for cls, ratio in class_groups.items():
        print(f"  {cls}: {ratio:.4f}")

    # Mask Area Statistics (Regression Target Proxy)
    mask_areas_nonzero = df[df["mask_area"] > 0]["mask_area"]
    print("\nMask Area Distribution (Non-zero masks only):")
    print(f"  Mean Area: {mask_areas_nonzero.mean():.4f} pixels")
    print(f"  Std Dev: {mask_areas_nonzero.std():.4f}")
    print(f"  Min: {mask_areas_nonzero.min()}")
    print(f"  Max: {mask_areas_nonzero.max()}")
    print(f"  Skewness: {mask_areas_nonzero.skew():.4f}")
    print(f"  Kurtosis: {mask_areas_nonzero.kurtosis():.4f}")

    # 3. Input Data Analysis (Image Modality)
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("=" * 30)

    # Dimensions from Metadata
    print("Image Dimensions:")
    print(f"  Widths: {df['img_width'].unique()}")
    print(f"  Heights: {df['img_height'].unique()}")

    # Aspect Ratios
    aspect_ratios = df["img_width"] / df["img_height"]
    print(f"  Mean Aspect Ratio: {aspect_ratios.mean():.4f}")

    # Physical Spacing
    print("\nPhysical Pixel Spacing (mm):")
    print(f"  Width Spacing Mean: {df['pixel_spacing_w'].mean():.4f}")
    print(f"  Height Spacing Mean: {df['pixel_spacing_h'].mean():.4f}")

    # Image Content Analysis (Sampling)
    # We sample unique images (ignoring class duplication)
    unique_images = df[["file_path"]].drop_duplicates()
    sample_size = min(500, len(unique_images))
    sample_paths = (
        unique_images["file_path"].sample(n=sample_size, random_state=SEED).values
    )

    pixel_vals = []
    channels = []
    dtypes = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if os.path.exists(full_path):
            # Read unchanged to detect 16-bit vs 8-bit
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Check channels
            if len(img.shape) == 2:
                channels.append(1)
            else:
                channels.append(img.shape[2])

            dtypes.append(img.dtype)

            # Flatten and collect stats (subsample pixels for speed)
            # Taking every 10th pixel to keep memory usage low while estimating stats
            pixel_vals.extend(img.flatten()[::100])

    pixel_vals = np.array(pixel_vals)

    print("\nSampled Image Content Stats (N=500):")
    unique_channels = np.unique(channels)
    print(f"  Channel Counts Found: {unique_channels}")
    unique_dtypes = np.unique(dtypes)
    print(f"  Data Types Found: {unique_dtypes}")

    if len(pixel_vals) > 0:
        print(f"  Global Pixel Mean: {np.mean(pixel_vals):.4f}")
        print(f"  Global Pixel Std: {np.std(pixel_vals):.4f}")
        print(f"  Global Pixel Min: {np.min(pixel_vals)}")
        print(f"  Global Pixel Max: {np.max(pixel_vals)}")

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Structured Relationships
    # Preparing features for correlation and importance
    # We use 'slice', 'day', 'pixel_spacing_w', 'img_width'

    # Create a subset for analysis
    analysis_df = df[
        [
            "slice",
            "day",
            "pixel_spacing_w",
            "img_width",
            "mask_area",
            "has_mask",
            "class",
        ]
    ].copy()

    # Encode class for correlation analysis
    le = LabelEncoder()
    analysis_df["class_encoded"] = le.fit_transform(analysis_df["class"])

    # Correlation Matrix
    print("Correlations with Mask Area (Numerical):")
    corr_matrix = analysis_df[
        ["slice", "day", "pixel_spacing_w", "img_width", "class_encoded", "mask_area"]
    ].corr()
    target_corr = corr_matrix["mask_area"].drop("mask_area")
    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.4f}")

    # Feature Importance (Random Forest)
    # Predict 'has_mask' based on metadata
    print("\nFeature Importance (Predicting 'has_mask' with Random Forest):")

    X = analysis_df[["slice", "day", "pixel_spacing_w", "img_width", "class_encoded"]]
    y = analysis_df["has_mask"]

    # Simple split not needed for feature importance check, fitting on sample
    sample_idx = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_sample, y_sample)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")

    # Meta-Feature Insight
    print("\nMeta-Feature Insights:")
    # Check if slice index correlates with mask presence (Anatomy check)
    # Bin slices into groups
    analysis_df["slice_bin"] = pd.cut(analysis_df["slice"], bins=5)
    bin_stats = analysis_df.groupby("slice_bin", observed=True)["has_mask"].mean()
    print("  Probability of Mask by Slice Position (Binned):")
    for interval, prob in bin_stats.items():
        print(f"    Slice {interval}: {prob:.4f}")


if __name__ == "__main__":
    main()
