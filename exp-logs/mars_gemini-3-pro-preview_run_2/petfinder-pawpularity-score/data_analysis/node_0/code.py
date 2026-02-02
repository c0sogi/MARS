import os
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train.csv"
INPUT_BASE_DIR = "./input"
SEED = 42
IMAGE_SAMPLE_SIZE = 1000  # Sample size for image analysis to ensure speed


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    return df


def analyze_target(df):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    target = df["Pawpularity"]

    print(f"Target Variable: Pawpularity")
    print(f"Type: Regression")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Skewness and Kurtosis
    target_skew = skew(target)
    target_kurt = kurtosis(target)

    print(f"Skewness: {target_skew:.4f} (Normal ~ 0)")
    print(f"Kurtosis: {target_kurt:.4f} (Normal ~ 0)")

    if abs(target_skew) > 1:
        print("Observation: Target distribution is highly skewed.")
    elif abs(target_skew) > 0.5:
        print("Observation: Target distribution is moderately skewed.")
    else:
        print("Observation: Target distribution is approximately symmetric.")
    print("")


def analyze_tabular(df):
    print("=" * 30)
    print("TABULAR INPUT DATA ANALYSIS")
    print("=" * 30)

    # Identify feature columns (exclude Id, file_path, Pawpularity)
    exclude_cols = ["Id", "file_path", "Pawpularity"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Number of Tabular Features: {len(feature_cols)}")

    # Missing Values
    missing = df[feature_cols].isnull().sum()
    total_missing = missing.sum()
    print(f"Total Missing Values in Features: {total_missing}")
    if total_missing > 0:
        print("Missing values per column:")
        print(missing[missing > 0])

    # Numerical/Categorical Analysis
    # In this dataset, features are binary (0/1), effectively categorical but treated as numeric
    print("\nBinary Feature Frequency (Percentage of '1's):")
    for col in feature_cols:
        mean_val = df[col].mean()
        print(f"  {col:<15}: {mean_val*100:.2f}%")

        # Check for rare labels
        if mean_val < 0.01 or mean_val > 0.99:
            print(f"    -> FLAG: Rare label distribution (<1% or >99%)")

    print("")
    return feature_cols


def analyze_images(df):
    print("=" * 30)
    print("IMAGE DATA ANALYSIS")
    print("=" * 30)

    # Sample images for analysis
    if len(df) > IMAGE_SAMPLE_SIZE:
        sample_df = df.sample(n=IMAGE_SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(f"Analyzing a sample of {len(sample_df)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Pixel stats accumulators
    pixel_means = []
    pixel_stds = []

    # Meta-features for relationship analysis later
    meta_features = {
        "img_width": [],
        "img_height": [],
        "img_aspect_ratio": [],
        "img_brightness": [],
        "target": [],
    }

    valid_samples = 0

    for idx, row in sample_df.iterrows():
        # Construct full path. Metadata path is relative to ./input
        # e.g., metadata says "train/abc.jpg", full path is "./input/train/abc.jpg"
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_BASE_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            img = cv2.imread(full_path)
            if img is None:
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channel_counts.append(c)

            # Calculate pixel stats (BGR to RGB doesn't affect global mean/std magnitude much, but we keep BGR)
            mean_val = np.mean(img)
            std_val = np.std(img)

            pixel_means.append(mean_val)
            pixel_stds.append(std_val)

            # Store for relationship analysis
            meta_features["img_width"].append(w)
            meta_features["img_height"].append(h)
            meta_features["img_aspect_ratio"].append(w / h)
            meta_features["img_brightness"].append(mean_val)
            meta_features["target"].append(row["Pawpularity"])

            valid_samples += 1

        except Exception as e:
            continue

    if valid_samples == 0:
        print("Error: No images could be loaded.")
        return None

    # Dimensions
    print("\nDimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.2f}, Std: {np.std(widths):.2f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.2f}, Std: {np.std(heights):.2f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )

    # Aspect Ratio
    print("\nAspect Ratios (Width/Height):")
    print(f"  Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")

    # Channels
    unique_channels = np.unique(channel_counts)
    print(f"\nChannels: {unique_channels} (3=RGB)")

    # Pixel Stats
    # Normalize to 0-1 range for reporting
    global_mean = np.mean(pixel_means) / 255.0
    global_std = np.mean(pixel_stds) / 255.0
    print("\nGlobal Pixel Statistics (Normalized 0-1):")
    print(f"  Mean: {global_mean:.4f}")
    print(f"  Std:  {global_std:.4f}")

    print("")
    return pd.DataFrame(meta_features)


def analyze_relationships(df, feature_cols, img_meta_df):
    print("=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    target = df["Pawpularity"]

    # 1. Structured Relationships (Tabular vs Target)
    print("Structured Data Relationships (Metadata vs Pawpularity):")

    # Correlations
    correlations = {}
    for col in feature_cols:
        corr, _ = pearsonr(df[col], target)
        correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n  Top 5 Correlated Binary Features (Pearson):")
    for name, val in sorted_corr[:5]:
        print(f"    {name:<15}: {val:.4f}")

    # Redundancy (Collinearity)
    print("\n  Redundancy Check (Correlation > 0.90):")
    found_redundancy = False
    corr_matrix = df[feature_cols].corr().abs()
    # Iterate over upper triangle
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.90:
                print(
                    f"    {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.4f}"
                )
                found_redundancy = True
    if not found_redundancy:
        print("    No highly collinear pairs found.")

    # Feature Importance (Random Forest)
    print("\n  Feature Importance (Random Forest):")
    X = df[feature_cols]
    y = target
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(indices))):
        print(f"    {feature_cols[indices[i]]:<15}: {importances[indices[i]]:.4f}")

    # 2. Unstructured Relationships (Image Meta vs Target)
    if img_meta_df is not None and len(img_meta_df) > 0:
        print("\nUnstructured Data Relationships (Image Stats vs Pawpularity):")

        # Correlation between image stats and target
        img_stats = ["img_width", "img_height", "img_aspect_ratio", "img_brightness"]

        for stat in img_stats:
            corr, _ = pearsonr(img_meta_df[stat], img_meta_df["target"])
            print(f"    Correlation {stat:<18} vs Target: {corr:.4f}")

        print("\n  Observation on Image Size:")
        # Check if larger images have higher scores
        large_imgs = img_meta_df[
            img_meta_df["img_width"] > img_meta_df["img_width"].median()
        ]
        small_imgs = img_meta_df[
            img_meta_df["img_width"] <= img_meta_df["img_width"].median()
        ]
        print(f"    Mean Target (Large Images): {large_imgs['target'].mean():.4f}")
        print(f"    Mean Target (Small Images): {small_imgs['target'].mean():.4f}")


def main():
    set_seed(SEED)

    try:
        df = load_data()

        analyze_target(df)
        feature_cols = analyze_tabular(df)
        img_meta_df = analyze_images(df)
        analyze_relationships(df, feature_cols, img_meta_df)

        print("\nEDA Completed Successfully.")

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
