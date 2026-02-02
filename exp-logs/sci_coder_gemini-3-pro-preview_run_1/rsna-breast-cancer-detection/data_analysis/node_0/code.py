import os
import sys
import numpy as np
import pandas as pd
import cv2
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE_IMG = 500  # Number of images to sample for pixel stats

# Set Seeds
random.seed(SEED)
np.random.seed(SEED)


def print_section(title):
    print(f"\n==== {title} ====")


def analyze_target(df):
    print_section("TARGET VARIABLE ANALYSIS")
    target = "cancer"

    # Distribution
    counts = df[target].value_counts()
    props = df[target].value_counts(normalize=True)

    print(f"Target Variable: '{target}'")
    print(f"Class 0 (Negative): {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 (Positive): {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    # Imbalance Ratio
    if counts.get(1, 0) > 0:
        ratio = counts.get(0, 0) / counts.get(1, 0)
        print(f"Imbalance Ratio (Neg/Pos): {ratio:.4f}")
    else:
        print("Imbalance Ratio: Infinite (No positive samples)")


def analyze_tabular(df):
    print_section("INPUT DATA ANALYSIS (TABULAR)")

    # Numerical Features
    num_cols = ["age"]
    print("--- Numerical Features ---")
    for col in num_cols:
        if col in df.columns:
            series = df[col]
            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()

            # Outliers (IQR)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)].count()

            print(f"Feature: {col}")
            print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
            print(f"  Outliers (IQR method): {outliers} ({outliers/len(df):.4f})")
        else:
            print(f"Feature: {col} (Not found)")

    # Categorical Features
    # Note: 'biopsy', 'invasive', 'BIRADS', 'difficult_negative_case' are auxiliary/labels, not inputs.
    # Inputs: site_id, laterality, view, implant, density, machine_id
    cat_cols = ["site_id", "laterality", "view", "implant", "density", "machine_id"]

    print("\n--- Categorical Features ---")
    for col in cat_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"Feature: {col}")
            print(f"  Cardinality: {unique_vals}")

            # Check for rare labels (< 1%)
            counts = df[col].value_counts(normalize=True)
            rare_labels = counts[counts < 0.01].index.tolist()
            if len(rare_labels) > 0:
                if len(rare_labels) > 5:
                    print(
                        f"  Rare labels (<1%): {len(rare_labels)} categories (e.g., {rare_labels[:3]}...)"
                    )
                else:
                    print(f"  Rare labels (<1%): {rare_labels}")
            else:
                print("  Rare labels (<1%): None")

            # Flag high cardinality
            if unique_vals > 50:
                print("  Flag: High Cardinality (>50)")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df[cat_cols + num_cols].isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"{col}: {count} missing ({count/len(df):.4f})")
    else:
        print("No missing values in input features.")


def try_load_image(path):
    # Attempt to load with cv2 (if it's a standard format disguised as dcm or if cv2 has dcm support)
    # Note: Standard cv2 does not support DICOM.
    # Since pydicom is not in the allowed list, we rely on cv2 or skip.
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except:
        pass
    return None


def analyze_images(df):
    print_section("INPUT DATA ANALYSIS (IMAGE)")

    # Check file existence and sizes
    sample_df = df.sample(n=min(SAMPLE_SIZE_IMG, len(df)), random_state=SEED).copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_means = []
    pixel_stds = []
    file_sizes = []

    load_success_count = 0

    for _, row in sample_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # File Size
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path) / (1024 * 1024))  # MB

        # Image Loading
        # Since pydicom is not available, we likely cannot load the pixel data.
        # We will try cv2 just in case, but expect failure.
        img = try_load_image(full_path)

        if img is not None:
            load_success_count += 1
            h, w = img.shape[:2]
            c = 1 if len(img.shape) == 2 else img.shape[2]

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channels.append(c)

            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))

    # Report File Sizes
    if file_sizes:
        print(f"Analyzed {len(file_sizes)} files for size.")
        print(
            f"File Size (MB): Mean={np.mean(file_sizes):.4f}, Std={np.std(file_sizes):.4f}, Max={np.max(file_sizes):.4f}"
        )

    # Report Image Stats
    if load_success_count > 0:
        print(f"\nAnalyzed {load_success_count} images for pixel stats.")
        print(f"Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}")
        print(f"Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}")
        print(f"Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}")

        unique_channels = np.unique(channels)
        print(f"Channels: {unique_channels}")

        print(f"Pixel Global Mean: {np.mean(pixel_means):.4f}")
        print(f"Pixel Global Std:  {np.mean(pixel_stds):.4f}")
    else:
        print(
            "\n[INFO] Could not load image pixel data (DICOM library likely missing or format unsupported by cv2)."
        )
        print("Skipping Dimensions and Pixel Stats analysis.")


def analyze_relationships(df):
    print_section("FEATURE/SIGNAL RELATIONSHIPS")

    # Prepare Data for Correlation/Importance
    # Features: age, site_id, laterality, view, implant, density, machine_id
    # Target: cancer

    model_df = df.copy()

    # 1. Preprocessing
    # Map Density (Ordinal)
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    model_df["density_encoded"] = model_df["density"].map(density_map)

    # Encode Categoricals
    le_cols = ["site_id", "laterality", "view", "machine_id"]
    for col in le_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype(str)  # Handle mixed types/NaNs as str
            le = LabelEncoder()
            model_df[f"{col}_encoded"] = le.fit_transform(model_df[col])

    # Select features for analysis
    features = [
        "age",
        "implant",
        "density_encoded",
        "site_id_encoded",
        "laterality_encoded",
        "view_encoded",
        "machine_id_encoded",
    ]
    features = [f for f in features if f in model_df.columns]

    # Impute NaNs for analysis
    imputer = SimpleImputer(strategy="mean")
    X = model_df[features]
    y = model_df["cancer"]

    # Handle NaNs
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    # 2. Correlation
    print("--- Correlation with Target (Pearson/Spearman) ---")
    corrs = []
    for col in features:
        corr = X_imputed[col].corr(y)
        corrs.append((col, corr))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs:
        print(f"{col}: {corr:.4f}")

    # 3. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_imputed, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top Features:")
    for i in range(min(5, len(features))):
        idx = indices[i]
        print(f"{i+1}. {features[idx]}: {importances[idx]:.4f}")

    # 4. Redundancy (Collinearity)
    print("\n--- Redundancy (High Correlation > 0.90) ---")
    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    if len(high_corr) > 0:
        for col in high_corr:
            # Find the feature it correlates with
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"{col} is highly correlated with {correlated_with}")
    else:
        print("No highly collinear pairs found.")

    # 5. Meta-Feature Relationships
    # Check if file size correlates with cancer (proxy for complexity/compression/info)
    # We need to map file sizes back to the dataframe
    print("\n--- Meta-Feature Relationships ---")
    # Sample a subset for speed
    meta_sample = df.sample(n=min(1000, len(df)), random_state=SEED).copy()
    meta_sample["file_size"] = meta_sample["file_path"].apply(
        lambda x: (
            os.path.getsize(os.path.join(INPUT_DIR, x))
            if os.path.exists(os.path.join(INPUT_DIR, x))
            else np.nan
        )
    )

    # Drop NaNs
    meta_sample = meta_sample.dropna(subset=["file_size"])

    if not meta_sample.empty:
        corr_size = meta_sample["file_size"].corr(meta_sample["cancer"])
        print(f"Correlation (File Size vs Cancer): {corr_size:.4f}")

        # Mean size by class
        mean_neg = meta_sample[meta_sample["cancer"] == 0]["file_size"].mean()
        mean_pos = meta_sample[meta_sample["cancer"] == 1]["file_size"].mean()
        print(f"Mean File Size (Negative): {mean_neg/(1024*1024):.4f} MB")
        print(f"Mean File Size (Positive): {mean_pos/(1024*1024):.4f} MB")


def main():
    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # 1. Target
    analyze_target(df_train)

    # 2. Tabular
    analyze_tabular(df_train)

    # 3. Image
    analyze_images(df_train)

    # 4. Relationships
    analyze_relationships(df_train)


if __name__ == "__main__":
    main()
