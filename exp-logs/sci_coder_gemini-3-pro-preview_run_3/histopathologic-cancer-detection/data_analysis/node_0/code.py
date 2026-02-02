import os
import sys
import numpy as np
import pandas as pd
import cv2
import random
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# --- Configuration & Setup ---
# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")

# Constants for Image Analysis
SAMPLE_SIZE = (
    5000  # Number of images to sample for pixel stats to ensure runtime safety
)


def main():
    print("SECTION 1: INITIALIZATION")
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    # Load training metadata
    df = pd.read_csv(TRAIN_META_PATH)
    print(f"Loaded training metadata with {len(df)} records.")

    # --- SECTION 2: TARGET VARIABLE ANALYSIS ---
    print("\nSECTION 2: TARGET VARIABLE ANALYSIS")

    target_col = "label"
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    # Distribution
    class_counts = df[target_col].value_counts()
    class_ratios = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Type: Classification (Binary)")
    print(f"Class Distribution:")
    for label, count in class_counts.items():
        ratio = class_ratios[label]
        print(f"  Class {label}: {count} samples ({ratio:.4f})")

    # Imbalance Check
    majority_class_ratio = class_ratios.max()
    print(f"Majority Class Ratio: {majority_class_ratio:.4f}")
    if majority_class_ratio > 0.6:
        print("  Note: Dataset shows moderate to high imbalance.")
    else:
        print("  Note: Dataset is relatively balanced.")

    # --- SECTION 3: INPUT DATA ANALYSIS (IMAGE MODALITY) ---
    print("\nSECTION 3: INPUT DATA ANALYSIS (IMAGE)")

    # We need to analyze images. Since there are ~140k images, reading all might be slow.
    # We will use a stratified sample.
    sample_df = df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 2), random_state=RANDOM_SEED)
    )
    print(
        f"Analyzing a stratified sample of {len(sample_df)} images for pixel statistics..."
    )

    widths = []
    heights = []
    channels = []
    aspect_ratios = []

    # Pixel stats accumulators (for approximate global mean/std)
    # Storing sum and sum_sq to compute mean/std
    pixel_sum = np.zeros(3)  # Assuming RGB max
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0

    # Meta-features for relationship analysis
    meta_features = {
        "brightness_mean": [],
        "contrast_std": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
        "sharpness": [],  # Laplacian variance
    }
    meta_labels = []

    missing_files = 0

    for idx, row in sample_df.iterrows():
        # Construct full path. Metadata contains relative path from input dir
        # Note: Metadata file_path is likely "train/id.tif"
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_files += 1
            continue

        # Read image
        try:
            img = cv2.imread(full_path)
            if img is None:
                missing_files += 1
                continue

            # OpenCV loads as BGR, convert to RGB for analysis
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            channels.append(c)
            aspect_ratios.append(w / h if h > 0 else 0)

            # Pixel Stats Accumulation
            # Normalize to 0-1 for calculation
            img_norm = img / 255.0
            pixel_sum += img_norm.sum(axis=(0, 1))
            pixel_sq_sum += (img_norm**2).sum(axis=(0, 1))
            total_pixels += h * w

            # Meta-feature extraction
            # Brightness: Mean of grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # Sharpness: Variance of Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            meta_features["brightness_mean"].append(brightness)
            meta_features["contrast_std"].append(contrast)
            meta_features["red_mean"].append(np.mean(img[:, :, 0]))
            meta_features["green_mean"].append(np.mean(img[:, :, 1]))
            meta_features["blue_mean"].append(np.mean(img[:, :, 2]))
            meta_features["sharpness"].append(sharpness)

            meta_labels.append(row[target_col])

        except Exception as e:
            # In case of corrupt file
            missing_files += 1

    if missing_files > 0:
        print(f"Warning: Could not read {missing_files} files.")

    if total_pixels == 0:
        print("Error: No valid images processed.")
        return

    # -- Dimensions Analysis --
    widths = np.array(widths)
    heights = np.array(heights)

    print("Dimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )

    unique_shapes = set(zip(widths, heights))
    if len(unique_shapes) == 1:
        print(f"  Consistency: All images have shape {list(unique_shapes)[0]}")
    else:
        print(
            f"  Consistency: Images have varying shapes. Found {len(unique_shapes)} unique shapes."
        )

    # -- Channels Analysis --
    unique_channels = np.unique(channels)
    print(f"Channels: {unique_channels}")
    if len(unique_channels) == 1 and unique_channels[0] == 3:
        print("  Format: RGB")
    elif len(unique_channels) == 1 and unique_channels[0] == 1:
        print("  Format: Grayscale")
    else:
        print("  Format: Mixed or Other")

    # -- Pixel Stats Analysis --
    # Global Mean and Std per channel
    global_mean = pixel_sum / total_pixels
    # std = sqrt(E[x^2] - (E[x])^2)
    global_sq_mean = pixel_sq_sum / total_pixels
    global_std = np.sqrt(global_sq_mean - global_mean**2)

    print("Pixel Statistics (Normalized 0-1):")
    print(f"  Red Channel:   Mean={global_mean[0]:.4f}, Std={global_std[0]:.4f}")
    print(f"  Green Channel: Mean={global_mean[1]:.4f}, Std={global_std[1]:.4f}")
    print(f"  Blue Channel:  Mean={global_mean[2]:.4f}, Std={global_std[2]:.4f}")

    # --- SECTION 4: FEATURE/SIGNAL RELATIONSHIPS ---
    print("\nSECTION 4: FEATURE/SIGNAL RELATIONSHIPS")
    print("Analyzing relationship between image meta-features and target label...")

    # Create DataFrame from meta-features
    meta_df = pd.DataFrame(meta_features)
    meta_df["label"] = meta_labels

    # 1. Correlation with Target
    # Since target is binary (0/1), Point-Biserial correlation is equivalent to Pearson
    correlations = meta_df.corr()["label"].drop("label")
    print("Correlation with Target (Pearson):")
    for feature, corr in correlations.sort_values(ascending=False).items():
        print(f"  {feature}: {corr:.4f}")

    # 2. Feature Importance (Random Forest)
    # We use these meta-features to see if simple stats can predict the tumor
    X_meta = meta_df.drop("label", axis=1)
    y_meta = meta_df["label"]

    rf = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1, max_depth=5
    )
    rf.fit(X_meta, y_meta)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop Meta-Feature Importance (Random Forest):")
    for i in range(min(5, len(indices))):
        feat_name = X_meta.columns[indices[i]]
        print(f"  {feat_name}: {importances[indices[i]]:.4f}")

    # 3. Class-Specific Distributions
    print("\nClass-Specific Meta-Feature Means:")
    grouped = meta_df.groupby("label").mean()
    for col in X_meta.columns:
        val_0 = grouped.loc[0, col]
        val_1 = grouped.loc[1, col]
        diff = val_1 - val_0
        print(
            f"  {col}: Class 0 Mean={val_0:.4f}, Class 1 Mean={val_1:.4f}, Diff={diff:.4f}"
        )

    # 4. Redundancy (Collinearity)
    print("\nFeature Redundancy (Correlation > 0.90):")
    corr_matrix = X_meta.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if not to_drop:
        print("  No highly collinear pairs found among meta-features.")
    else:
        for col in to_drop:
            correlated_cols = upper.index[upper[col] > 0.90].tolist()
            for row in correlated_cols:
                print(f"  {row} - {col}: {upper.loc[row, col]:.4f}")


if __name__ == "__main__":
    main()
