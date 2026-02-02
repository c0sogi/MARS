import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
import warnings

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE_IMG = 200  # Number of images to sample for detailed analysis

# Set Seeds
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_target(df, target_col):
    print_section("Target Variable Analysis")
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return

    counts = df[target_col].value_counts()
    total = len(df)
    print(f"Target: {target_col}")
    print(f"Distribution:\n{counts}")

    # Class balance
    for cls, count in counts.items():
        print(f"Class {cls}: {count} ({count/total:.4%})")

    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        print(f"Imbalance Ratio: 1:{ratio:.2f}")


def analyze_tabular(df, target_col):
    print_section("Tabular Data Analysis")

    # Identify feature types
    # Features available in Test: site_id, laterality, view, age, implant, machine_id
    # Features only in Train: density, biopsy, invasive, BIRADS, difficult_negative_case

    numerical_cols = ["age"]
    categorical_cols = [
        "site_id",
        "laterality",
        "view",
        "implant",
        "machine_id",
        "density",
        "BIRADS",
    ]

    # Numerical Analysis
    print("--- Numerical Features ---")
    for col in numerical_cols:
        if col in df.columns:
            desc = df[col].describe()
            q1 = desc["25%"]
            q3 = desc["75%"]
            iqr = q3 - q1
            outliers = (
                (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
            ).sum()
            print(f"Feature: {col}")
            print(f"  Mean: {desc['mean']:.4f}, Std: {desc['std']:.4f}")
            print(f"  Min: {desc['min']:.4f}, Max: {desc['max']:.4f}")
            print(f"  Outliers (IQR method): {outliers} ({outliers/len(df):.2%})")
            print(f"  Missing: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")

    # Categorical Analysis
    print("\n--- Categorical Features ---")
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"Feature: {col}")
            print(f"  Cardinality: {unique_vals}")
            print(f"  Missing: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")

            if unique_vals > 50:
                print(
                    f"  > High cardinality column. Top 5 values: {df[col].value_counts().head(5).index.tolist()}"
                )
            else:
                # Check for rare labels
                counts = df[col].value_counts(normalize=True)
                rare = counts[counts < 0.01]
                if not rare.empty:
                    print(f"  > Rare labels (<1%): {list(rare.index)}")


def analyze_images(df):
    print_section("Image Data Analysis")

    # Check for pydicom
    try:
        import pydicom

        HAS_PYDICOM = True
    except ImportError:
        HAS_PYDICOM = False
        print("Note: 'pydicom' library not found. Skipping pixel-level analysis.")

    # Sample images
    sample_df = df.sample(n=min(SAMPLE_SIZE_IMG, len(df)), random_state=SEED)

    widths = []
    heights = []
    ratios = []
    file_sizes = []
    pixel_means = []
    pixel_stds = []

    print(f"Analyzing sample of {len(sample_df)} images...")

    for _, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # File size
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        file_sizes.append(size_mb)

        if HAS_PYDICOM:
            try:
                # Only read headers first to be fast
                dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                h = float(dcm.Rows)
                w = float(dcm.Columns)
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)

                # Attempt pixel stats on a smaller sub-sample (first 20) to save time
                if len(pixel_means) < 20:
                    try:
                        dcm_pixels = pydicom.dcmread(full_path)
                        arr = dcm_pixels.pixel_array.astype(float)
                        pixel_means.append(np.mean(arr))
                        pixel_stds.append(np.std(arr))
                    except Exception:
                        pass  # Pixel data might be compressed (JPEG2000) and require missing libs
            except Exception:
                pass

    # Report Stats
    print("\n--- File Properties ---")
    if file_sizes:
        print(
            f"File Size (MB): Mean={np.mean(file_sizes):.4f}, Std={np.std(file_sizes):.4f}, Max={np.max(file_sizes):.4f}"
        )

    if HAS_PYDICOM and widths:
        print("\n--- Image Dimensions ---")
        print(
            f"Width:  Mean={np.mean(widths):.1f}, Std={np.std(widths):.1f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Height: Mean={np.mean(heights):.1f}, Std={np.std(heights):.1f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(f"Aspect Ratio: Mean={np.mean(ratios):.4f}")

        if pixel_means:
            print("\n--- Pixel Intensity (Subsample) ---")
            print(f"Global Mean: {np.mean(pixel_means):.4f}")
            print(f"Global Std:  {np.mean(pixel_stds):.4f}")
    elif HAS_PYDICOM:
        print(
            "Could not extract dimensions from sampled DICOM files (possibly due to compression or read errors)."
        )

    return file_sizes


def analyze_relationships(df, target_col, file_sizes_sample=None):
    print_section("Feature/Signal Relationships")

    # 1. Structured Relationships
    print("--- Structured Data Relationships ---")

    # Numerical Correlation
    if "age" in df.columns:
        # Point Biserial Correlation for Age vs Cancer
        clean_df = df.dropna(subset=["age", target_col])
        corr, pval = stats.pointbiserialr(clean_df[target_col], clean_df["age"])
        print(f"Correlation (Age vs Cancer): {corr:.4f} (p-value: {pval:.4f})")

    # Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    # Use features available in Test set
    features = ["site_id", "laterality", "view", "age", "implant", "machine_id"]
    features = [f for f in features if f in df.columns]

    if features:
        X = df[features].copy()
        y = df[target_col].copy()

        # Preprocessing
        # Handle Categoricals
        cat_cols = X.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))

        # Handle NaNs
        imp = SimpleImputer(strategy="mean")
        X = imp.fit_transform(X)

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
        )
        rf.fit(X, y)

        # Report
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Top Features predicting Cancer:")
        for i in range(min(5, len(features))):
            print(f"  {i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

    # Redundancy Check
    print("\n--- Redundancy (Collinearity) ---")
    # Check correlation between numerical/encoded features
    if features:
        df_corr = pd.DataFrame(X, columns=features)
        corr_matrix = df_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
        if high_corr:
            print(f"Highly collinear features (>0.90): {high_corr}")
        else:
            print("No highly collinear features found (>0.90).")

    # 2. Unstructured/Meta Relationships
    print("\n--- Meta-Feature Relationships ---")

    # Relationship between Density and Cancer (Train only, but informative)
    if "density" in df.columns:
        print("Cancer Rate by Breast Density:")
        density_stats = df.groupby("density")[target_col].mean()
        print(density_stats)

    # Relationship between View and Cancer
    if "view" in df.columns:
        print("\nCancer Rate by View:")
        view_stats = df.groupby("view")[target_col].mean().sort_values(ascending=False)
        print(view_stats.head(5))


def main():
    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Run Analysis
    analyze_target(df, "cancer")
    analyze_tabular(df, "cancer")
    file_sizes = analyze_images(df)
    analyze_relationships(df, "cancer", file_sizes)


if __name__ == "__main__":
    main()
