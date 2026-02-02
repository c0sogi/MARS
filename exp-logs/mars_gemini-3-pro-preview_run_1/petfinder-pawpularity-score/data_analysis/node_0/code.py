import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df, target_col):
    print("2. TARGET VARIABLE ANALYSIS")
    target = df[target_col]
    print(f"Target Variable: {target_col}")

    # Distribution stats
    print(f"Mean: {target.mean():.4f}")
    print(f"Std:  {target.std():.4f}")
    print(f"Min:  {target.min():.4f}")
    print(f"Max:  {target.max():.4f}")

    # Normality
    s = target.skew()
    k = target.kurt()
    print(f"Skewness: {s:.4f}")
    print(f"Kurtosis: {k:.4f}")

    # Outliers (IQR Method)
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target[(target < lower_bound) | (target > upper_bound)]
    print(f"Outlier Count (IQR Method): {len(outliers)}")
    print("-" * 30)


def analyze_tabular(df, feature_cols):
    print("3. INPUT DATA ANALYSIS (TABULAR)")
    print("Analyzing binary metadata features...")

    # Numerical/Binary stats
    # Since they are binary, mean represents the frequency of the positive class
    stats = df[feature_cols].agg(["mean", "min", "max"])
    print(stats.T.to_string(float_format="{:.4f}".format))

    # Missing values
    missing = df[feature_cols].isnull().sum()
    total_missing = missing.sum()
    print(f"\nTotal Missing Values in Features: {total_missing}")

    # Cardinality / Rare Labels
    print("\nCardinality & Rare Label Check:")
    for col in feature_cols:
        counts = df[col].value_counts(normalize=True)
        if len(counts) > 50:
            print(f"Column '{col}' has high cardinality: {len(counts)}")

        # Check for rare labels (< 1%)
        rare = counts[counts < 0.01]
        if not rare.empty:
            print(f"Column '{col}' has rare labels: {rare.index.tolist()}")

    print("-" * 30)


def analyze_images(df, path_col, input_root):
    print("3. INPUT DATA ANALYSIS (IMAGE)")

    # Sample for efficiency
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    print(f"Analyzing a sample of {sample_size} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_indices = []

    for idx, row in sample_df.iterrows():
        full_path = os.path.join(input_root, row[path_col])
        if not os.path.exists(full_path):
            continue

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # CV2 reads as BGR, convert to RGB for stats
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels.append(c)

        # Normalize to 0-1 for stats
        img_norm = img / 255.0
        pixel_sum += img_norm.sum(axis=(0, 1))
        pixel_sq_sum += (img_norm**2).sum(axis=(0, 1))
        pixel_count += w * h

        valid_indices.append(idx)

    # Filter sample_df to only valid images for later correlation
    sample_df = sample_df.loc[valid_indices]
    sample_df["img_width"] = widths
    sample_df["img_height"] = heights
    sample_df["img_aspect"] = aspect_ratios

    # Dimensions
    print("\nDimensions:")
    print(
        f"Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # Channels
    unique_channels = np.unique(channels)
    print(f"Channel Counts: {unique_channels}")

    # Pixel Stats
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))
        print("\nGlobal Pixel Stats (RGB, 0-1 range):")
        print(f"Mean: {global_mean}")
        print(f"Std:  {global_std}")

    print("-" * 30)
    return sample_df


def analyze_relationships(df, feature_cols, target_col, img_sample_df):
    print("4. FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Structured Relationships
    print("Structured (Tabular) Relationships:")

    # Correlation
    correlations = df[feature_cols + [target_col]].corr(method="pearson")
    target_corr = correlations[target_col].drop(target_col)
    print("\nFeature-Target Correlation (Pearson):")
    print(
        target_corr.sort_values(ascending=False).to_string(float_format="{:.4f}".format)
    )

    # Redundancy
    print("\nRedundancy (Collinear Pairs > 0.90):")
    found_redundancy = False
    corr_matrix = df[feature_cols].corr().abs()

    # Iterate to print pairs
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if val > 0.90:
                print(f"{feature_cols[i]} - {feature_cols[j]}: {val:.4f}")
                found_redundancy = True
    if not found_redundancy:
        print("None found.")

    # Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")
    X = df[feature_cols]
    y = df[target_col]
    rf = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print(
        importances.sort_values(ascending=False)
        .head(5)
        .to_string(float_format="{:.4f}".format)
    )

    # 2. Unstructured Relationships
    print("\nUnstructured (Meta-Feature) Relationships:")
    if img_sample_df is not None and not img_sample_df.empty:
        meta_features = ["img_width", "img_height", "img_aspect"]
        # Calculate correlation on the sample
        meta_corr = (
            img_sample_df[meta_features + [target_col]]
            .corr()[target_col]
            .drop(target_col)
        )
        print("Correlation between Image Meta-Features and Target:")
        print(meta_corr.to_string(float_format="{:.4f}".format))
    else:
        print("No image data available for meta-feature analysis.")

    print("-" * 30)


def main():
    set_seed(42)

    INPUT_ROOT = "./input"
    METADATA_PATH = "./metadata/train.csv"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    print("1. LOADING DATA")
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df)} rows from {METADATA_PATH}")
    print("-" * 30)

    # Define columns
    target_col = "Pawpularity"
    # Features are all columns except Id, Pawpularity, file_path
    feature_cols = [
        c for c in df.columns if c not in ["Id", "Pawpularity", "file_path"]
    ]

    # 2. Target Analysis
    analyze_target(df, target_col)

    # 3. Input Analysis
    # Tabular
    analyze_tabular(df, feature_cols)
    # Image
    img_sample_df = analyze_images(df, "file_path", INPUT_ROOT)

    # 4. Relationships
    analyze_relationships(df, feature_cols, target_col, img_sample_df)


if __name__ == "__main__":
    main()
