import os
import glob
import pandas as pd
import numpy as np
import random
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Configuration & Setup
# ==========================================
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*40}")
    print(f" {title.upper()}")
    print(f"{'='*40}")


def attempt_read_dicom(path):
    """
    Attempts to read a DICOM file using available libraries.
    Returns (image_array, method_used) or (None, None).
    """
    # Attempt 1: pydicom (Standard, but might not be in the allowed list)
    try:
        import pydicom

        ds = pydicom.dcmread(path)
        return ds.pixel_array, "pydicom"
    except ImportError:
        pass
    except Exception:
        pass

    # Attempt 2: OpenCV (Sometimes works for specific formats, usually fails on raw DICOM)
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img, "cv2"
    except Exception:
        pass

    return None, None


def analyze_target(df):
    print_section("Target Variable Analysis")
    target_col = "MGMT_value"

    # Distribution
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)

    print(f"Target: {target_col} (Binary Classification)")
    print(f"Class 0 Count: {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 Count: {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    # Imbalance check
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"Class Balance Ratio (0/1): {ratio:.4f}")
    if 0.4 < ratio < 2.5:
        print("Status: Balanced")
    else:
        print("Status: Imbalanced")


def extract_filesystem_metadata(df):
    """
    Iterates through the training set to extract file counts and sizes.
    This serves as 'Image Analysis' when pixel reading is expensive or restricted.
    """
    print_section("Input Data Analysis (Structure & Metadata)")

    modalities = ["flair", "t1w", "t1wce", "t2w"]
    stats = []

    print(f"Processing {len(df)} subjects to extract file system metadata...")

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        subject_stats = {"BraTS21ID": sid, "MGMT_value": row["MGMT_value"]}

        for mod in modalities:
            # Construct path relative to input dir
            # metadata contains relative paths like 'train/00000/FLAIR'
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if os.path.exists(full_path):
                files = os.listdir(full_path)
                dcm_files = [f for f in files if f.endswith(".dcm")]

                # Count
                count = len(dcm_files)
                subject_stats[f"{mod}_count"] = count

                # Size (taking average size of first 10 files to save time, or all if small)
                sample_files = dcm_files[:10]
                sizes = []
                for f in sample_files:
                    try:
                        sizes.append(os.path.getsize(os.path.join(full_path, f)))
                    except OSError:
                        pass

                avg_size = np.mean(sizes) if sizes else 0
                subject_stats[f"{mod}_avg_size_bytes"] = avg_size
            else:
                subject_stats[f"{mod}_count"] = 0
                subject_stats[f"{mod}_avg_size_bytes"] = 0

        stats.append(subject_stats)

    df_stats = pd.DataFrame(stats)

    # Report Statistics
    print("\n--- Modality File Counts (Slice Depth) ---")
    for mod in modalities:
        col = f"{mod}_count"
        print(
            f"{mod.upper()}: Mean={df_stats[col].mean():.2f}, Min={df_stats[col].min()}, Max={df_stats[col].max()}"
        )

    print("\n--- Modality File Sizes (Bytes) ---")
    for mod in modalities:
        col = f"{mod}_avg_size_bytes"
        print(
            f"{mod.upper()}: Mean={df_stats[col].mean():.2f}, Std={df_stats[col].std():.2f}"
        )

    return df_stats


def analyze_image_pixels(df):
    print_section("Image Data Analysis (Pixel Stats)")

    # Sample 1 subject, 1 image from each modality
    sample_row = df.iloc[0]
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    successful_read = False

    for mod in modalities:
        rel_path = sample_row[f"{mod}_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
        if not files:
            continue

        # Pick middle file
        img_path = os.path.join(full_path, files[len(files) // 2])

        img, method = attempt_read_dicom(img_path)

        if img is not None:
            successful_read = True
            print(f"Successfully read {mod.upper()} image using {method}.")
            print(f"Dimensions: {img.shape}")
            print(f"Dtype: {img.dtype}")
            print(f"Min Value: {img.min()}, Max Value: {img.max()}")
            print(f"Mean Value: {img.mean():.4f}, Std Dev: {img.std():.4f}")

            # Check channels
            if len(img.shape) == 2:
                print("Channels: 1 (Grayscale)")
            else:
                print(f"Channels: {img.shape[2]}")
            break  # Just analyze one successful modality to prove capability

    if not successful_read:
        print(
            "WARNING: Could not read DICOM pixel data with installed packages (cv2/pydicom)."
        )
        print(
            "Pixel-level statistics (mean, std, dimensions) cannot be computed directly."
        )
        print(
            "Analysis will rely on File System Metadata (counts and sizes) as proxies."
        )


def analyze_relationships(df_stats):
    print_section("Feature/Signal Relationships")

    target = "MGMT_value"
    features = [c for c in df_stats.columns if c not in ["BraTS21ID", "MGMT_value"]]

    # 1. Correlation
    print("\n--- Correlation with Target (Pearson) ---")
    corrs = df_stats[features + [target]].corr()[target].drop(target)
    print(corrs.sort_values(ascending=False))

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    X = df_stats[features].fillna(0)
    y = df_stats[target]

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED)
    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("Top 5 Meta-Features predicting MGMT_value:")
    print(importances.head(5))

    # 3. Collinearity
    print("\n--- Redundancy Check (Correlation > 0.90) ---")
    corr_matrix = df_stats[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        print(f"Highly correlated features found: {to_drop}")
        print("Consider removing these to reduce noise.")
    else:
        print("No highly collinear pairs found among meta-features.")


def main():
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Target Analysis
    analyze_target(df)

    # 3. Image Pixel Analysis (Sample)
    analyze_image_pixels(df)

    # 4. Structure/Metadata Analysis (Full Dataset)
    # We extract features like 'number of slices' or 'file size'
    # which act as proxies for image properties.
    df_stats = extract_filesystem_metadata(df)

    # 5. Relationship Analysis
    analyze_relationships(df_stats)

    print_section("EDA Complete")


if __name__ == "__main__":
    main()
