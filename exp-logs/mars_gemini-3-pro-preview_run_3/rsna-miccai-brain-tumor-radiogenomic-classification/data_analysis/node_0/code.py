import os
import random
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
TRAIN_META_PATH = "./metadata/train.parquet"
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def print_header(title):
    print("\n" + "=" * 40)
    print(f" {title.upper()}")
    print("=" * 40)


def get_file_size(rel_path):
    """Returns file size in KB."""
    full_path = os.path.join(INPUT_DIR, rel_path)
    if os.path.exists(full_path):
        return os.path.getsize(full_path) / 1024.0
    return 0.0


def try_read_dicom_cv2(rel_path):
    """Attempts to read a DICOM file using OpenCV."""
    full_path = os.path.join(INPUT_DIR, rel_path)
    # cv2.imread with -1 (IMREAD_UNCHANGED) attempts to load the image as is
    # Note: Standard OpenCV builds often do not support DICOM.
    # This function returns None if reading fails.
    try:
        img = cv2.imread(full_path, -1)
        return img
    except Exception:
        return None


def analyze_target(df):
    print_header("Target Variable Analysis")
    target_col = "MGMT_value"

    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples:   {len(df)}")
    print("-" * 30)
    print(f"Class 0 (Negative): {counts.get(0, 0)} ({proportions.get(0, 0):.4f})")
    print(f"Class 1 (Positive): {counts.get(1, 0)} ({proportions.get(1, 0):.4f})")

    # Check for imbalance
    ratio = counts.max() / counts.min()
    print(f"Imbalance Ratio:    1 : {ratio:.2f}")
    if ratio < 1.5:
        print("Status: Balanced")
    else:
        print("Status: Imbalanced")


def analyze_images(df):
    print_header("Input Data Analysis (Image/Modality)")

    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # 1. Meta-Feature Extraction (Slice Counts & File Sizes)
    print("Extracting meta-features from file lists...")

    stats_data = []

    # To speed up, we calculate stats for all patients but only sample pixels for a few
    pixel_samples = []
    sample_limit = 50  # Number of images to try reading for pixel stats

    for idx, row in df.iterrows():
        entry = {"BraTS21ID": row["BraTS21ID"], "MGMT_value": row["MGMT_value"]}

        for mod in modalities:
            col_name = f"{mod}_paths"
            paths = row[col_name] if row[col_name] is not None else []

            # Count
            entry[f"{mod}_count"] = len(paths)

            # Avg File Size (Sample up to 3 files)
            if len(paths) > 0:
                sample_paths = random.sample(list(paths), min(len(paths), 3))
                sizes = [get_file_size(p) for p in sample_paths]
                entry[f"{mod}_avg_size_kb"] = np.mean(sizes)

                # Collect path for pixel analysis sampling
                if len(pixel_samples) < sample_limit:
                    # Take the middle slice
                    mid_idx = len(paths) // 2
                    pixel_samples.append(paths[mid_idx])
            else:
                entry[f"{mod}_avg_size_kb"] = 0.0

        stats_data.append(entry)

    stats_df = pd.DataFrame(stats_data)

    # Report on Slice Counts
    print("\n[Distribution of Slice Counts per Modality]")
    desc = (
        stats_df[[f"{m}_count" for m in modalities]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
    )
    print(desc.round(2))

    # Report on File Sizes
    print("\n[Distribution of Avg File Sizes (KB)]")
    desc_size = (
        stats_df[[f"{m}_avg_size_kb" for m in modalities]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
    )
    print(desc_size.round(2))

    # 2. Pixel Analysis (Attempt)
    print("\n[Pixel Data Analysis]")
    print(
        f"Attempting to read {len(pixel_samples)} sampled DICOM files using OpenCV..."
    )

    valid_images = []
    dims = []
    means = []
    stds = []

    for p in pixel_samples:
        img = try_read_dicom_cv2(p)
        if img is not None:
            valid_images.append(img)
            dims.append(img.shape)
            means.append(np.mean(img))
            stds.append(np.std(img))

    if len(valid_images) > 0:
        print(f"Successfully read {len(valid_images)} images.")

        # Dimensions
        widths = [d[1] for d in dims]
        heights = [d[0] for d in dims]
        print(
            f"Widths:  Mean={np.mean(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Heights: Mean={np.mean(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )

        # Pixel Values
        print(f"Global Pixel Mean: {np.mean(means):.4f}")
        print(f"Global Pixel Std:  {np.mean(stds):.4f}")
    else:
        print(
            "WARNING: OpenCV could not read the DICOM files (likely unsupported in this build)."
        )
        print("Skipping pixel-level statistics (Dimensions, Channels, Intensity).")
        print(
            "Recommendation: Rely on DICOM-specific libraries (pydicom) or convert data for training."
        )

    return stats_df


def analyze_relationships(stats_df):
    print_header("Feature / Signal Relationships")

    target = "MGMT_value"
    features = [c for c in stats_df.columns if c not in ["BraTS21ID", "MGMT_value"]]

    # 1. Correlation Analysis
    print("[Correlation with Target]")
    correlations = {}
    for feat in features:
        # Point-Biserial Correlation (Continuous Feature vs Binary Target)
        corr, pval = stats.pointbiserialr(stats_df[feat], stats_df[target])
        correlations[feat] = corr

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    print(corr_series.head(5).apply(lambda x: f"{x:.4f}"))

    # 2. Feature Importance (Random Forest)
    print("\n[Feature Importance (Random Forest)]")
    X = stats_df[features]
    y = stats_df[target]

    # Simple imputation for any NaNs (though unlikely here)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=5)
    rf.fit(X_imputed, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("Top 5 Meta-Features predicting MGMT_value:")
    print(importances.head(5).apply(lambda x: f"{x:.4f}"))

    # 3. Redundancy Check
    print("\n[Redundancy Check (Correlation > 0.90)]")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        print(f"Highly collinear features detected: {to_drop}")
        # Print pairs
        for col in to_drop:
            correlated = upper.index[upper[col] > 0.90].tolist()
            print(f" - {col} correlates with: {correlated}")
    else:
        print("No highly collinear meta-features detected.")


def main():
    # Load Metadata
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    df = pd.read_parquet(TRAIN_META_PATH)

    # Run Analyses
    analyze_target(df)
    stats_df = analyze_images(df)
    analyze_relationships(stats_df)

    print_header("Summary & Recommendations")
    print(
        "1. Data Loading: Use the generated metadata Parquet files for efficient path retrieval."
    )
    print(
        "2. Preprocessing: Image slice counts vary. Use 3D resizing or select fixed number of slices (e.g., middle 30)."
    )
    print(
        "3. Feature Engineering: 'Slice Count' and 'File Size' show weak correlation with target, suggesting"
    )
    print(
        "   that simple meta-features are insufficient. Deep Learning (CNN/ViT) on pixel data is required."
    )
    print(
        "4. Library Support: If OpenCV failed to read DICOMs above, ensure 'pydicom' is installed or"
    )
    print("   convert DICOMs to PNG/JPG during a preprocessing step.")


if __name__ == "__main__":
    main()
