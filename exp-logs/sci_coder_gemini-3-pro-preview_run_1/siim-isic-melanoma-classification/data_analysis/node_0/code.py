import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from scipy.stats import skew, kurtosis, pearsonr

# --- Configuration & Constants ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
IMAGE_SAMPLE_SIZE = 1000  # Number of images to sample for pixel/dimension stats

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def set_pandas_display():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)


def load_data():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(METADATA_PATH)
    return df


def analyze_target(df):
    print("=== TARGET VARIABLE ANALYSIS ===")
    target_col = "target"

    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: '{target_col}'")
    print(f"Total Samples: {total}")

    # Distribution
    for label, count in counts.items():
        ratio = count / total
        print(f"Class {label}: {count} ({ratio:.4%})")

    # Imbalance
    if len(counts) == 2:
        # Assuming binary classification 0/1
        maj_class = counts.idxmax()
        min_class = counts.idxmin()
        ratio = counts[maj_class] / counts[min_class]
        print(f"Class Balance Ratio (Majority/Minority): {ratio:.4f}")
    else:
        print("Multiclass target detected.")
    print("-" * 30)


def analyze_tabular(df):
    print("=== INPUT DATA ANALYSIS (TABULAR) ===")

    # Identify column types
    # Based on dataset description:
    # Numerical: age_approx
    # Categorical: sex, anatom_site_general_challenge, diagnosis, patient_id
    # Ignored: image_name, file_path, benign_malignant (proxy for target)

    numerical_cols = ["age_approx"]
    categorical_cols = ["sex", "anatom_site_general_challenge", "diagnosis"]

    # 1. Numerical Analysis
    print("--- Numerical Features ---")
    for col in numerical_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()

        # Outliers via IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        print(f"Feature: {col}")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
        print(
            f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(series):.4%})"
        )

    # 2. Categorical Analysis
    print("\n--- Categorical Features ---")
    for col in categorical_cols:
        if col not in df.columns:
            continue

        series = df[col].astype(
            str
        )  # Handle NaNs as 'nan' string for cardinality check first
        unique_vals = series.nunique()

        print(f"Feature: {col}")
        print(f"  Cardinality: {unique_vals}")

        if unique_vals > 50:
            print("  Flag: High Cardinality (> 50 categories)")

        # Rare labels
        counts = series.value_counts(normalize=True)
        rare = counts[counts < 0.01]
        if not rare.empty:
            print(f"  Rare Labels (<1% freq): {len(rare)} categories")

    # 3. Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        for col, count in missing.items():
            ratio = count / len(df)
            print(f"  {col}: {count} missing ({ratio:.4%})")
    print("-" * 30)


def analyze_images(df):
    print("=== INPUT DATA ANALYSIS (IMAGE) ===")

    # Sample images to save time
    sample_df = df.sample(n=min(len(df), IMAGE_SAMPLE_SIZE), random_state=SEED)

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    print(f"Sampling {len(sample_df)} images for analysis...")

    valid_samples = 0

    for _, row in sample_df.iterrows():
        # Construct full path. Metadata path is relative to input dir root?
        # Based on verification script: "jpeg/train/{x}.jpg" inside INPUT_DIR
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        img = cv2.imread(full_path)
        if img is None:
            continue

        # OpenCV loads as BGR, convert to RGB for standard stats
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channel_counts.append(c)

        # Pixel stats
        # Normalize to 0-1 for calculation
        img_norm = img / 255.0
        pixel_sum += img_norm.sum(axis=(0, 1))
        pixel_sq_sum += (img_norm**2).sum(axis=(0, 1))
        pixel_count += h * w

        valid_samples += 1

    if valid_samples == 0:
        print("No valid images found in sample.")
        return

    # Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(aspect_ratios)

    print("--- Dimensions ---")
    print(
        f"Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(f"Aspect Ratio: Mean={ratios.mean():.4f}, Std={ratios.std():.4f}")

    # Channels
    unique_channels = np.unique(channel_counts)
    print(f"Channels: {unique_channels}")

    # Pixel Stats
    global_mean = pixel_sum / pixel_count
    global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))

    print("--- Pixel Statistics (Normalized 0-1) ---")
    print(
        f"Mean (R, G, B): {global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}"
    )
    print(
        f"Std  (R, G, B): {global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}"
    )
    print("-" * 30)

    # Return sampled stats for meta-feature analysis
    return pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "target": sample_df["target"].values[
                :valid_samples
            ],  # Align with valid samples
        }
    )


def analyze_relationships(df, img_stats_df):
    print("=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Prepare Data for Correlation/Importance
    # Select relevant columns
    features = ["age_approx", "sex", "anatom_site_general_challenge"]
    target = "target"

    # Create a working copy
    work_df = df[features + [target]].copy()

    # Preprocessing
    # 1. Impute Numerical
    num_imputer = SimpleImputer(strategy="mean")
    work_df["age_approx"] = num_imputer.fit_transform(work_df[["age_approx"]])

    # 2. Encode Categorical
    le_dict = {}
    for col in ["sex", "anatom_site_general_challenge"]:
        # Fill NaN with 'missing'
        work_df[col] = work_df[col].fillna("missing").astype(str)
        le = LabelEncoder()
        work_df[col] = le.fit_transform(work_df[col])
        le_dict[col] = le

    # 1. Structured Relationships
    print("--- Structured Data Relationships ---")

    # Correlation (Numerical vs Target)
    corr_age, _ = pearsonr(work_df["age_approx"], work_df["target"])
    print(f"Correlation (Age vs Target): {corr_age:.4f}")

    # Mutual Information (Categorical vs Target)
    for col in ["sex", "anatom_site_general_challenge"]:
        mi = mutual_info_score(work_df[col], work_df["target"])
        print(f"Mutual Information ({col} vs Target): {mi:.4f}")

    # Feature Importance (Random Forest)
    X = work_df[features]
    y = work_df[target]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop Features (Random Forest Importance):")
    for f in range(len(features)):
        idx = indices[f]
        print(f"  {features[idx]}: {importances[idx]:.4f}")

    # Redundancy Check
    print("\nRedundancy (Correlation > 0.90):")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if not high_corr:
        print("  No highly collinear pairs found among basic metadata features.")
    else:
        for col in high_corr:
            print(f"  {col} is highly correlated with other features.")

    # 2. Unstructured Relationships
    print("\n--- Unstructured (Meta-Feature) Relationships ---")
    if img_stats_df is not None and not img_stats_df.empty:
        # Check correlation between image size and target
        # Is malignancy associated with image size?
        corr_w, _ = pearsonr(img_stats_df["width"], img_stats_df["target"])
        corr_h, _ = pearsonr(img_stats_df["height"], img_stats_df["target"])

        print(f"Correlation (Image Width vs Target): {corr_w:.4f}")
        print(f"Correlation (Image Height vs Target): {corr_h:.4f}")

        # Check if mean size differs by class
        grp = img_stats_df.groupby("target")[["width", "height"]].mean()
        print("\nMean Image Dimensions by Class:")
        print(grp)
    else:
        print("Skipping meta-feature analysis (no image stats available).")

    print("-" * 30)


def main():
    set_pandas_display()

    # 1. Load Data
    df = load_data()

    # 2. Target Analysis
    analyze_target(df)

    # 3. Tabular Analysis
    analyze_tabular(df)

    # 4. Image Analysis
    img_stats_df = analyze_images(df)

    # 5. Relationships
    analyze_relationships(df, img_stats_df)


if __name__ == "__main__":
    main()
