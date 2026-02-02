import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42

# Set seeds
np.random.seed(SEED)


def set_seed(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def run_eda():
    print("STARTING EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 40)

    # 1. Load Data
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # Construct full image paths
    # The metadata contains relative paths like 'images/1.jpg'
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    print(f"Loaded Training Data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Target Variable Analysis
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "species"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
    else:
        class_counts = df[target_col].value_counts()
        n_classes = len(class_counts)
        min_count = class_counts.min()
        max_count = class_counts.max()
        balance_ratio = max_count / min_count if min_count > 0 else 0

        print(f"Target Variable: '{target_col}' (Categorical)")
        print(f"Number of Classes: {n_classes}")
        print(
            f"Class Distribution: Min samples = {min_count}, Max samples = {max_count}"
        )
        print(f"Class Balance Ratio (Max/Min): {balance_ratio:.4f}")

        if balance_ratio > 1.5:
            print("Observation: The dataset shows class imbalance.")
        else:
            print("Observation: The dataset is relatively balanced.")

    # 3. Input Data Analysis (Tabular)
    print("\nINPUT DATA ANALYSIS: TABULAR")
    print("-" * 30)

    # Identify feature columns (exclude id, species, file_path, full_path)
    exclude_cols = ["id", "species", "file_path", "full_path"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Check for missing values
    missing_counts = df[feature_cols].isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Missing Values: {total_missing} total across all feature columns.")
    if total_missing > 0:
        cols_with_missing = missing_counts[missing_counts > 0]
        print(f"Columns with missing values: {cols_with_missing.index.tolist()}")

    # Numerical Stats Summary
    # Since there are 192 features, we summarize the distribution of their stats
    stats = df[feature_cols].describe().T
    stats["iqr"] = stats["75%"] - stats["25%"]

    # Outlier detection (1.5 * IQR)
    outlier_counts = {}
    for col in feature_cols:
        q1 = stats.loc[col, "25%"]
        q3 = stats.loc[col, "75%"]
        iqr = stats.loc[col, "iqr"]
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = n_outliers

    total_outliers = sum(outlier_counts.values())
    avg_outliers_per_col = np.mean(list(outlier_counts.values()))

    print(f"Total Numerical Features: {len(feature_cols)}")
    print(
        f"Global Mean of Features: {stats['mean'].mean():.4f} (Range: {stats['mean'].min():.4f} to {stats['mean'].max():.4f})"
    )
    print(f"Global Std Dev of Features: {stats['std'].mean():.4f}")
    print(f"Outliers (IQR Method): {total_outliers} total detected across all columns.")
    print(f"Average Outliers per Column: {avg_outliers_per_col:.2f}")

    # 4. Input Data Analysis (Image)
    print("\nINPUT DATA ANALYSIS: IMAGE")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_sums = 0.0
    pixel_sq_sums = 0.0
    pixel_count = 0

    # Process images
    # We'll process all images since N=~700 is small
    valid_images = 0
    for idx, row in df.iterrows():
        path = row["full_path"]
        if os.path.exists(path):
            # Read as is to check channels
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            c = 1 if len(img.shape) == 2 else img.shape[2]

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channels.append(c)

            # For pixel stats, convert to float and normalize to 0-1 for calculation
            # Assuming standard 8-bit images
            img_flat = img.flatten() / 255.0
            pixel_sums += np.sum(img_flat)
            pixel_sq_sums += np.sum(img_flat**2)
            pixel_count += len(img_flat)
            valid_images += 1
        else:
            pass  # File missing handled in metadata validation usually

    if valid_images > 0:
        print(f"Analyzed {valid_images} images.")
        print(
            f"Widths: Mean={np.mean(widths):.2f}, Std={np.std(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Heights: Mean={np.mean(heights):.2f}, Std={np.std(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
        )

        unique_channels = np.unique(channels)
        print(
            f"Channel Counts Distribution: {dict(zip(*np.unique(channels, return_counts=True)))}"
        )

        # Global Pixel Stats
        global_mean = pixel_sums / pixel_count
        global_std = np.sqrt((pixel_sq_sums / pixel_count) - (global_mean**2))
        print(f"Global Pixel Mean (normalized 0-1): {global_mean:.4f}")
        print(f"Global Pixel Std (normalized 0-1): {global_std:.4f}")
    else:
        print("No valid images found for analysis.")

    # 5. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # A. Structured Relationships (Tabular)
    if len(feature_cols) > 1:
        # Correlation
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        print(f"High Correlation (> 0.90): Found {len(to_drop)} redundant features.")
        if len(to_drop) > 0:
            print(f"Sample redundant features: {to_drop[:5]} ...")

        # Feature Importance (Random Forest)
        le = LabelEncoder()
        y_enc = le.fit_transform(df[target_col])
        X = df[feature_cols].fillna(0)  # Handle NaNs if any

        rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
        rf.fit(X, y_enc)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 5 Features (Random Forest Importance):")
        for i in range(5):
            print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

    # B. Unstructured (Meta-Feature) Relationships
    # Does image area correlate with species?
    if valid_images > 0:
        # Add area to dataframe for those we calculated
        # We need to map back carefully. Since we iterated rows:
        # Let's just re-calculate for the dataframe subset
        areas = []
        species_list = []

        for idx, row in df.iterrows():
            path = row["full_path"]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    areas.append(w * h)
                    species_list.append(row[target_col])

        meta_df = pd.DataFrame({"area": areas, "species": species_list})

        # ANOVA test: Is there a significant difference in image area between species?
        groups = [group["area"].values for name, group in meta_df.groupby("species")]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            print(f"\nMeta-Feature Analysis (Image Area vs Species):")
            print(f"One-way ANOVA F-statistic: {f_stat:.4f}")
            print(f"P-value: {p_val:.4e}")
            if p_val < 0.05:
                print(
                    "Observation: Image dimensions significantly vary across different species."
                )
            else:
                print(
                    "Observation: Image dimensions do not significantly vary across species."
                )


if __name__ == "__main__":
    run_eda()
