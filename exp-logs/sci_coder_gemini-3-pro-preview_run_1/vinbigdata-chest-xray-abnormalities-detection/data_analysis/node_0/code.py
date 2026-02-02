import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Configuration ---
METADATA_PATH = "./metadata/train_meta.csv"
INPUT_DIR = "./input"
SEED = 42

# Set seeds
np.random.seed(SEED)


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")


def print_sub_header(title):
    print(f"\n--- {title} ---")


def analyze_target(df):
    print_header("TARGET VARIABLE ANALYSIS")

    # Distribution of Classes
    print_sub_header("Class Distribution")
    class_counts = df["class_name"].value_counts()
    class_props = df["class_name"].value_counts(normalize=True)

    print(f"{'Class Name':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 45)
    for name, count in class_counts.items():
        prop = class_props[name]
        print(f"{name:<25} {count:<10} {prop:.4%}")

    # Imbalance
    print_sub_header("Class Imbalance")
    max_class = class_counts.iloc[0]
    min_class = class_counts.iloc[-1]
    ratio = max_class / min_class
    print(f"Most frequent class: {class_counts.index[0]} ({max_class})")
    print(f"Least frequent class: {class_counts.index[-1]} ({min_class})")
    print(f"Imbalance Ratio (Max/Min): {ratio:.4f}")

    # No Finding vs Finding
    print_sub_header("Findings vs No Findings")
    n_no_finding = df[df["class_id"] == 14].shape[0]
    n_finding = df[df["class_id"] != 14].shape[0]
    total = len(df)
    print(f"No Finding (Class 14): {n_no_finding} ({n_no_finding/total:.4%})")
    print(f"Findings (Classes 0-13): {n_finding} ({n_finding/total:.4%})")


def analyze_tabular(df):
    print_header("INPUT DATA ANALYSIS (TABULAR/ANNOTATIONS)")

    # Filter out Class 14 for geometric analysis (they are 1x1 pixel placeholders)
    df_findings = df[df["class_id"] != 14].copy()

    if len(df_findings) == 0:
        print("No findings available for tabular analysis.")
        return

    # Derive Features
    df_findings["width"] = df_findings["x_max"] - df_findings["x_min"]
    df_findings["height"] = df_findings["y_max"] - df_findings["y_min"]
    df_findings["area"] = df_findings["width"] * df_findings["height"]
    df_findings["aspect_ratio"] = df_findings["width"] / df_findings["height"]

    # Numerical Analysis
    cols_to_analyze = ["width", "height", "area", "aspect_ratio"]
    print_sub_header("Bounding Box Geometry Statistics (Excluding 'No finding')")

    for col in cols_to_analyze:
        stats = df_findings[col].describe()

        # Outlier Detection (IQR)
        Q1 = stats["25%"]
        Q3 = stats["75%"]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_findings[
            (df_findings[col] < lower_bound) | (df_findings[col] > upper_bound)
        ]

        print(f"\nFeature: {col}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(
            f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df_findings):.2%})"
        )

    # Categorical Analysis (Radiologist ID)
    print_sub_header("Categorical Analysis: Radiologist ID")
    rad_counts = df["rad_id"].value_counts()
    print(f"Total Unique Radiologists: {df['rad_id'].nunique()}")
    print(f"Top 5 Radiologists by Annotation Count:")
    print(rad_counts.head(5).to_string())

    # Missing Values
    print_sub_header("Missing Values")
    missing = df.isnull().sum()
    print(
        missing[missing > 0].to_string()
        if missing.sum() > 0
        else "No missing values found in metadata."
    )


def analyze_images(df):
    print_header("INPUT DATA ANALYSIS (IMAGE)")

    # Attempt to import pydicom (often required for DICOM) or use CV2
    # Note: The environment description suggests restricted packages.
    # We will try robustly and report if we cannot read images.

    try:
        import pydicom

        HAS_PYDICOM = True
    except ImportError:
        HAS_PYDICOM = False

    try:
        import cv2

        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    # Sample images for analysis
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=SEED)

    widths = []
    heights = []
    pixel_means = []
    pixel_stds = []
    channels = []

    successful_reads = 0

    print(f"Attempting to analyze a sample of {sample_size} images...")

    for idx, row in sample_df.iterrows():
        # Construct full path. Metadata path is relative to input dir structure
        # row['file_path'] is like "train/xxxx.dicom"
        # We need to prepend "./input"
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        img = None

        # Strategy 1: Pydicom
        if HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(full_path)
                img = ds.pixel_array
            except Exception:
                pass

        # Strategy 2: OpenCV (if pydicom failed or not present)
        if img is None and HAS_CV2:
            try:
                # cv2.imread might fail on DICOM depending on build
                img = cv2.imread(full_path, -1)
            except Exception:
                pass

        if img is not None:
            successful_reads += 1
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

            # Check channels
            if len(img.shape) == 2:
                channels.append(1)
            else:
                channels.append(img.shape[2])

            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))

    if successful_reads == 0:
        print("\n[WARNING] Could not read any DICOM images.")
        print("Reasons might include:")
        print("1. 'pydicom' library is not installed.")
        print("2. 'opencv-python' installed does not support DICOM I/O.")
        print("Skipping pixel-level analysis.")
        return

    # Report Stats
    print_sub_header("Image Dimensions")
    print(f"Mean Width:  {np.mean(widths):.4f} (Std: {np.std(widths):.4f})")
    print(f"Mean Height: {np.mean(heights):.4f} (Std: {np.std(heights):.4f})")

    aspect_ratios = np.array(widths) / np.array(heights)
    print(f"Mean Aspect Ratio: {np.mean(aspect_ratios):.4f}")

    print_sub_header("Channel Distribution")
    unique_channels, counts = np.unique(channels, return_counts=True)
    for c, count in zip(unique_channels, counts):
        print(f"Channels {c}: {count} images")

    print_sub_header("Pixel Intensity Statistics")
    print(f"Global Mean Pixel Value: {np.mean(pixel_means):.4f}")
    print(f"Global Std Pixel Value:  {np.mean(pixel_stds):.4f}")


def analyze_relationships(df):
    print_header("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Structured Relationships (Correlations)
    df_findings = df[df["class_id"] != 14].copy()
    if len(df_findings) > 0:
        df_findings["width"] = df_findings["x_max"] - df_findings["x_min"]
        df_findings["height"] = df_findings["y_max"] - df_findings["y_min"]
        df_findings["area"] = df_findings["width"] * df_findings["height"]

        print_sub_header("Correlation Matrix (Bounding Box Features)")
        corr = df_findings[["x_min", "y_min", "width", "height", "area"]].corr()
        print(corr.round(4))

        # Redundancy Check
        print("\nChecking for Redundancy (Correlation > 0.90):")
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.90:
                    high_corr.append(
                        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                    )

        if high_corr:
            for c1, c2, val in high_corr:
                print(f"  {c1} and {c2}: {val:.4f}")
        else:
            print("  No highly collinear pairs found.")

    # 2. Feature Importance (Random Forest)
    # Task: Predict 'class_id' based on bounding box location/shape
    if len(df_findings) > 100:
        print_sub_header("Feature Importance (Predicting Class from BBox)")

        X = df_findings[["x_min", "y_min", "x_max", "y_max", "width", "height", "area"]]
        y = df_findings["class_id"]

        # Encode target if necessary (it is already int, but let's be safe)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Train small RF
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED)
        rf.fit(X, y_enc)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Top Features for Class Prediction:")
        for f in range(len(X.columns)):
            print(
                f"  {f+1}. {X.columns[indices[f]]:<10} ({importances[indices[f]]:.4f})"
            )

    # 3. Meta-Feature Relationship: Box Area vs Class
    if len(df_findings) > 0:
        print_sub_header("Relationship: Mean Bounding Box Area per Class")
        area_by_class = (
            df_findings.groupby("class_name")["area"]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"{'Class Name':<25} {'Mean Area':<15}")
        print("-" * 40)
        for name, area in area_by_class.items():
            print(f"{name:<25} {area:.4f}")


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Target Analysis
    analyze_target(df)

    # 3. Tabular Analysis
    analyze_tabular(df)

    # 4. Image Analysis
    analyze_images(df)

    # 5. Relationships
    analyze_relationships(df)

    print("\nEDA Completed.")


if __name__ == "__main__":
    main()
