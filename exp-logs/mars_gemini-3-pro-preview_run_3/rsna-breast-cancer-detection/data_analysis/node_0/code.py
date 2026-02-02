import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from scipy.stats import skew, kurtosis

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def load_dicom_safe(path):
    """
    Attempts to load a DICOM file.
    Returns (pixel_array, metadata_dict) or None if failed/library missing.
    """
    try:
        import pydicom

        ds = pydicom.dcmread(path)
        return ds.pixel_array, ds
    except ImportError:
        return None
    except Exception:
        return None


# ------------------------------------------------------------------------------
# Main Analysis Logic
# ------------------------------------------------------------------------------
def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # --------------------------------------------------------------------------
    # 2. Target Variable Analysis
    # --------------------------------------------------------------------------
    print_section("Target Variable Analysis")
    target_col = "cancer"

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset.")
    else:
        counts = df[target_col].value_counts()
        total = len(df)
        ratios = df[target_col].value_counts(normalize=True)

        print(f"Target Variable: {target_col}")
        print(f"Distribution:\n{counts.to_string()}")
        print(f"Ratios:\n{ratios.to_string()}")

        # Imbalance
        if len(counts) == 2:
            maj_class = counts.idxmax()
            min_class = counts.idxmin()
            imbalance_ratio = counts[maj_class] / counts[min_class]
            print(f"Class Imbalance Ratio (Maj/Min): {imbalance_ratio:.4f}")
        else:
            print("Target is not binary.")

    # --------------------------------------------------------------------------
    # 3. Input Data Analysis (Tabular)
    # --------------------------------------------------------------------------
    print_section("Tabular Data Analysis")

    # Identify column types
    # Based on dataset description
    num_cols = ["age"]
    cat_cols = [
        "site_id",
        "laterality",
        "view",
        "implant",
        "density",
        "machine_id",
        "biopsy",
        "invasive",
        "BIRADS",
    ]

    # Numerical Analysis
    print("--- Numerical Features ---")
    for col in num_cols:
        if col in df.columns:
            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

            print(f"Feature: {col}")
            print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
            print(f"  Min: {series.min():.4f}, Max: {series.max():.4f}")
            print(f"  Outliers (IQR method): {outliers} ({outliers/len(series):.2%})")

    # Categorical Analysis
    print("\n--- Categorical Features ---")
    for col in cat_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"Feature: {col} | Cardinality: {unique_vals}")

            if unique_vals > 50:
                print(f"  (High cardinality, showing top 5)")
                print(f"  {df[col].value_counts().head(5).to_dict()}")
            else:
                # Check for rare labels (< 1%)
                vc_norm = df[col].value_counts(normalize=True)
                rare = vc_norm[vc_norm < 0.01]
                if not rare.empty:
                    print(f"  Rare labels (<1%): {list(rare.index)}")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        for col, count in missing.items():
            print(f"{col}: {count} NaNs ({count/len(df):.2%})")

    # --------------------------------------------------------------------------
    # 3b. Input Data Analysis (Image)
    # --------------------------------------------------------------------------
    print_section("Image Data Analysis")

    # Construct full paths
    # The metadata contains relative paths in 'file_path'
    # We need to prepend INPUT_DIR
    image_paths = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x)).tolist()

    # Sample for analysis to save time
    SAMPLE_SIZE = 200
    if len(image_paths) > SAMPLE_SIZE:
        sampled_paths = np.random.choice(image_paths, SAMPLE_SIZE, replace=False)
    else:
        sampled_paths = image_paths

    # Check if pydicom is importable
    try:
        import pydicom

        HAS_PYDICOM = True
    except ImportError:
        HAS_PYDICOM = False
        print(
            "Note: 'pydicom' not installed. Skipping pixel-level analysis. Analyzing file sizes only."
        )

    widths = []
    heights = []
    aspect_ratios = []
    means = []
    stds = []
    file_sizes = []

    valid_images_count = 0

    for p in sampled_paths:
        if os.path.exists(p):
            # File size analysis (always possible)
            file_sizes.append(os.path.getsize(p))

            if HAS_PYDICOM:
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=False)
                    # Handle pixel data
                    if "PixelData" in ds:
                        arr = ds.pixel_array
                        h, w = arr.shape
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(w / h if h > 0 else 0)
                        means.append(np.mean(arr))
                        stds.append(np.std(arr))
                        valid_images_count += 1
                except Exception:
                    # Some DICOMs (e.g. JPEG2000) might fail without specific plugins
                    continue

    # Report File Size Stats
    if file_sizes:
        fs_series = pd.Series(file_sizes) / 1024 / 1024  # MB
        print(
            f"File Size (MB) - Mean: {fs_series.mean():.4f}, Std: {fs_series.std():.4f}, Max: {fs_series.max():.4f}"
        )

    if HAS_PYDICOM and valid_images_count > 0:
        print(f"Analyzed {valid_images_count} images (Sampled).")

        # Dimensions
        w_series = pd.Series(widths)
        h_series = pd.Series(heights)
        ar_series = pd.Series(aspect_ratios)

        print(
            f"Widths  - Mean: {w_series.mean():.4f}, Std: {w_series.std():.4f}, Min: {w_series.min()}, Max: {w_series.max()}"
        )
        print(
            f"Heights - Mean: {h_series.mean():.4f}, Std: {h_series.std():.4f}, Min: {h_series.min()}, Max: {h_series.max()}"
        )
        print(
            f"Aspect Ratios - Mean: {ar_series.mean():.4f}, Std: {ar_series.std():.4f}"
        )

        # Pixel Stats
        m_series = pd.Series(means)
        s_series = pd.Series(stds)
        print(
            f"Pixel Mean Intensity - Global Mean: {m_series.mean():.4f}, Std: {m_series.std():.4f}"
        )
        print(f"Pixel Std Intensity  - Global Mean: {s_series.mean():.4f}")

    elif HAS_PYDICOM:
        print(
            "Could not successfully read pixel data from sampled images (likely compression format issues)."
        )

    # --------------------------------------------------------------------------
    # 4. Feature/Signal Relationships
    # --------------------------------------------------------------------------
    print_section("Feature Relationships")

    # Prepare DataFrame for correlation/importance
    # We exclude 'biopsy', 'invasive', 'BIRADS', 'difficult_negative_case' from importance
    # because they are train-only labels/proxies for the target.
    # We include 'density' as it's a valid biological feature, though technically train-only in this dataset description.

    analysis_df = df.copy()

    # Encode Categoricals
    encoders = {}
    # Features to analyze against target
    features_to_check = [
        "site_id",
        "laterality",
        "view",
        "implant",
        "density",
        "machine_id",
        "age",
    ]

    # Handle NaNs for analysis
    for col in features_to_check:
        if col in analysis_df.columns:
            if analysis_df[col].dtype == "object":
                analysis_df[col] = analysis_df[col].fillna("Missing")
                le = LabelEncoder()
                analysis_df[col] = le.fit_transform(analysis_df[col].astype(str))
                encoders[col] = le
            else:
                analysis_df[col] = analysis_df[col].fillna(analysis_df[col].median())

    # 1. Correlation (Numerical/Encoded vs Target)
    print("--- Correlations with Target (cancer) ---")
    correlations = {}
    for col in features_to_check:
        if col in analysis_df.columns:
            corr = analysis_df[col].corr(analysis_df["cancer"])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_corr:
        print(f"{name}: {val:.4f}")

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    # Use only features available at inference (plus density for insight)
    rf_cols = [c for c in features_to_check if c in analysis_df.columns]

    X = analysis_df[rf_cols]
    y = analysis_df["cancer"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features:")
    for i in range(min(5, len(rf_cols))):
        print(f"{i+1}. {rf_cols[indices[i]]} ({importances[indices[i]]:.4f})")

    # 3. Redundancy (Collinearity)
    print("\n--- Redundancy Check (Correlation > 0.90) ---")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if high_corr:
        for col in high_corr:
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"{col} is highly correlated with: {correlated_with}")
    else:
        print("No highly collinear features found among selected predictors.")

    # 4. Meta-Feature Relationships
    print("\n--- Meta-Feature Relationships ---")
    # Age vs Cancer
    if "age" in df.columns:
        mean_age_pos = df[df["cancer"] == 1]["age"].mean()
        mean_age_neg = df[df["cancer"] == 0]["age"].mean()
        print(f"Mean Age (Cancer=1): {mean_age_pos:.4f}")
        print(f"Mean Age (Cancer=0): {mean_age_neg:.4f}")

    # Density vs Cancer (if density available)
    if "density" in df.columns:
        # Density is usually A, B, C, D. Let's see prevalence per density.
        print("Cancer Prevalence by Density:")
        density_stats = df.groupby("density")["cancer"].mean()
        print(density_stats.to_string(float_format="{:.4f}".format))

    # Implant vs Cancer
    if "implant" in df.columns:
        print("Cancer Prevalence by Implant Status:")
        implant_stats = df.groupby("implant")["cancer"].mean()
        print(implant_stats.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
