import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 2000  # Number of images to sample for pixel-level analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


def main():
    print("SECTION 1: DATA INTEGRITY & LOADING")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load metadata
    df = pd.read_csv(METADATA_PATH)
    print(f"Metadata loaded. Total training samples: {len(df)}")

    # Ensure we are only looking at training data (redundant given file name, but good practice)
    # The metadata generation script ensures this file only contains training IDs.

    print("\nSECTION 2: TARGET VARIABLE ANALYSIS")
    # Distribution
    target_counts = df["label"].value_counts()
    total_samples = len(df)

    print("Target Variable: 'label'")
    print("Distribution:")
    for label, count in target_counts.items():
        print(f"  Class {label}: {count} ({count/total_samples*100:.4f}%)")

    # Imbalance check
    class_ratio = target_counts.min() / target_counts.max()
    print(f"Class Balance Ratio (Min/Max): {class_ratio:.4f}")
    if class_ratio < 0.1:
        print("  NOTE: Severe class imbalance detected.")
    else:
        print("  NOTE: Classes are relatively balanced.")

    print("\nSECTION 3: INPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Sampling for Image Analysis
    # We sample to ensure the code runs within the time limit while providing good estimates.
    # We stratify by 'algo' to ensure coverage of Cover and all 3 stego algorithms.
    sample_df = df.groupby("algo", group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 4), random_state=SEED)
    )

    print(f"Analyzing a stratified sample of {len(sample_df)} images...")

    # Storage for stats
    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_means = []
    pixel_stds = []
    file_sizes = []

    # Meta-features for relationship analysis
    meta_features = []

    for idx, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # File Size
        try:
            f_size = os.path.getsize(full_path)
        except OSError:
            continue  # Skip if file read error

        file_sizes.append(f_size)

        # Image Load
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Dimensions
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h != 0 else 0)
        channels.append(c)

        # Pixel Stats (Normalize to 0-1 for reporting global stats)
        # Using float32 for precision
        img_norm = img.astype(np.float32) / 255.0
        p_mean = np.mean(img_norm)
        p_std = np.std(img_norm)

        pixel_means.append(p_mean)
        pixel_stds.append(p_std)

        # Store for meta-feature analysis
        meta_features.append(
            {
                "label": row["label"],
                "algo": row["algo"],
                "file_size_bytes": f_size,
                "width": w,
                "height": h,
                "mean_intensity": p_mean,
                "std_intensity": p_std,
            }
        )

    # Convert lists to numpy arrays for calculation
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels = np.array(channels)
    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)
    file_sizes = np.array(file_sizes)

    # Reporting
    print("Dimensions:")
    print(
        f"  Widths: Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    print("Channels:")
    unique_channels, counts_channels = np.unique(channels, return_counts=True)
    for c, count in zip(unique_channels, counts_channels):
        print(f"  {c} Channels: {count} images")

    print("Pixel Stats (Normalized 0-1):")
    print(f"  Global Mean Intensity: {np.mean(pixel_means):.4f}")
    print(f"  Global Std Deviation: {np.mean(pixel_stds):.4f}")

    print("\nSECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    # Create DataFrame from collected meta-features
    meta_df = pd.DataFrame(meta_features)

    # 1. Unstructured (Meta-Feature) Relationships
    print("Meta-Feature Analysis (Sampled Data):")

    # File Size vs Label
    # Are stego images larger?
    mean_size_cover = meta_df[meta_df["label"] == 0]["file_size_bytes"].mean()
    mean_size_stego = meta_df[meta_df["label"] == 1]["file_size_bytes"].mean()
    print(f"  Avg File Size (Cover): {mean_size_cover:.4f} bytes")
    print(f"  Avg File Size (Stego): {mean_size_stego:.4f} bytes")

    # Correlation
    # Encode algo for correlation matrix
    le = LabelEncoder()
    meta_df["algo_enc"] = le.fit_transform(meta_df["algo"])

    # Calculate correlations with label
    correlations = meta_df[
        [
            "label",
            "file_size_bytes",
            "width",
            "height",
            "mean_intensity",
            "std_intensity",
        ]
    ].corr()
    label_corr = correlations["label"].drop("label")
    print("  Correlations with Target (Label):")
    print(label_corr.to_string())

    # 2. Structured Importance (Lightweight Random Forest)
    # We use the meta-features to see if they hold any predictive power
    print("\nMeta-Feature Importance (Random Forest):")
    X = meta_df[
        ["file_size_bytes", "width", "height", "mean_intensity", "std_intensity"]
    ]
    y = meta_df["label"]

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Top Features predicting Label from Metadata:")
    for f in range(len(X.columns)):
        print(f"  {f+1}. {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

    # 3. Redundancy
    print("\nFeature Redundancy (Correlation > 0.90):")
    high_corr_pairs = []
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for column in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, column] > 0.90:
                high_corr_pairs.append((idx, column, upper.loc[idx, column]))

    if not high_corr_pairs:
        print("  No highly collinear meta-features found.")
    else:
        for p in high_corr_pairs:
            print(f"  {p[0]} - {p[1]}: {p[2]:.4f}")


if __name__ == "__main__":
    main()
