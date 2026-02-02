import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
RANDOM_STATE = 42


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)


def load_data():
    """
    Loads the training metadata and the corresponding raw JSON data.
    Filters the raw JSON to only include records present in the training split.
    """
    # Load metadata
    df_meta = pd.read_csv(TRAIN_META_PATH)

    # Load raw json
    with open(TRAIN_JSON_PATH, "r") as f:
        raw_data = json.load(f)

    # The metadata contains 'sample_index' which corresponds to the index in the raw list
    # We extract only the training samples
    train_indices = df_meta["sample_index"].values
    train_ids = set(df_meta["id"].values)

    # Filter raw data
    # We can index directly since we have the indices
    train_samples = [raw_data[i] for i in train_indices]

    # Verify alignment (optional but good for sanity)
    # The metadata generation script ensures indices are correct.

    return df_meta, train_samples


def analyze_target(df):
    """
    Analyzes the distribution of the target variable 'is_iceberg'.
    """
    print("TARGET VARIABLE ANALYSIS")
    print("========================")

    counts = df["is_iceberg"].value_counts()
    props = df["is_iceberg"].value_counts(normalize=True)

    print(f"Target Variable: is_iceberg")
    print(f"Class 0 (Ship):    {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 (Iceberg): {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    ratio = counts.get(0, 0) / max(counts.get(1, 0), 1)
    print(f"Imbalance Ratio (0/1): {ratio:.4f}")
    print("")


def analyze_images(samples):
    """
    Analyzes the image data (band_1 and band_2).
    """
    print("INPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("====================================")

    # Extract bands
    # Each band is a list of 5625 floats (75x75)
    b1 = np.array([s["band_1"] for s in samples])
    b2 = np.array([s["band_2"] for s in samples])

    # Dimensions
    n_samples = len(samples)
    width = 75
    height = 75
    n_pixels = width * height

    # Check if all lengths are consistent
    consistent_len = all(len(s["band_1"]) == n_pixels for s in samples)

    print(f"Image Dimensions: {width}x{height}")
    print(f"Consistent Dimensions: {consistent_len}")
    print(f"Channel Count: 2 (Band 1: HH, Band 2: HV)")
    print("")

    # Pixel Stats
    print("Pixel Statistics (Global):")

    # Band 1
    b1_mean = np.mean(b1)
    b1_std = np.std(b1)
    b1_min = np.min(b1)
    b1_max = np.max(b1)

    print(
        f"Band 1 (HH) - Mean: {b1_mean:.4f}, Std: {b1_std:.4f}, Min: {b1_min:.4f}, Max: {b1_max:.4f}"
    )

    # Band 2
    b2_mean = np.mean(b2)
    b2_std = np.std(b2)
    b2_min = np.min(b2)
    b2_max = np.max(b2)

    print(
        f"Band 2 (HV) - Mean: {b2_mean:.4f}, Std: {b2_std:.4f}, Min: {b2_min:.4f}, Max: {b2_max:.4f}"
    )
    print("")


def analyze_tabular(df):
    """
    Analyzes the tabular metadata (incidence angle).
    """
    print("INPUT DATA ANALYSIS (TABULAR MODALITY)")
    print("======================================")

    # inc_angle
    col = "inc_angle"
    print(f"Feature: {col}")

    # Missing values
    n_missing = df[col].isna().sum()
    pct_missing = n_missing / len(df)
    print(f"Missing Values: {n_missing} ({pct_missing:.4f})")

    # Stats (excluding NaN)
    valid_data = df[col].dropna()
    if len(valid_data) > 0:
        print(f"Mean: {valid_data.mean():.4f}")
        print(f"Std:  {valid_data.std():.4f}")
        print(f"Min:  {valid_data.min():.4f}")
        print(f"Max:  {valid_data.max():.4f}")

        # Outliers (IQR)
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        print(f"Outliers (IQR method): {len(outliers)}")
    else:
        print("No valid numerical data available.")
    print("")


def analyze_relationships(df, samples):
    """
    Analyzes relationships between features and the target.
    Extracts meta-features from images to perform correlation analysis.
    """
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("============================")

    # 1. Feature Engineering (Meta-features)
    # Calculate mean, std, min, max for each image to treat as tabular features
    b1 = np.array([s["band_1"] for s in samples])
    b2 = np.array([s["band_2"] for s in samples])

    features = pd.DataFrame()
    features["b1_mean"] = np.mean(b1, axis=1)
    features["b1_std"] = np.std(b1, axis=1)
    features["b1_min"] = np.min(b1, axis=1)
    features["b1_max"] = np.max(b1, axis=1)

    features["b2_mean"] = np.mean(b2, axis=1)
    features["b2_std"] = np.std(b2, axis=1)
    features["b2_min"] = np.min(b2, axis=1)
    features["b2_max"] = np.max(b2, axis=1)

    # Add inc_angle and target
    features["inc_angle"] = df["inc_angle"].values
    features["target"] = df["is_iceberg"].values

    # Handle missing inc_angle for correlation/model
    # We use a copy for analysis to not mutate original
    df_analysis = features.copy()

    # Impute missing inc_angle with mean for this analysis
    imputer = SimpleImputer(strategy="mean")
    df_analysis["inc_angle"] = imputer.fit_transform(df_analysis[["inc_angle"]])

    # 2. Correlation
    print("Correlations with Target (Pearson):")
    correlations = (
        df_analysis.corr()["target"]
        .drop("target")
        .sort_values(key=abs, ascending=False)
    )
    print(correlations.apply(lambda x: f"{x:.4f}"))
    print("")

    # 3. Redundancy (Collinearity)
    print("High Collinearity Pairs (Correlation > 0.90):")
    corr_matrix = df_analysis.drop("target", axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    found_collinear = False
    for col in upper.columns:
        for row in upper.index:
            if row != col and upper.loc[row, col] > 0.90:
                # Avoid printing duplicates (row, col) vs (col, row) handled by upper triangle
                if row < col:  # simple string comparison to print unique pairs
                    # Actually upper triangle logic handles uniqueness of pairs
                    pass

    # Iterate through upper triangle to print specific pairs
    pairs = []
    for r in range(len(upper.columns)):
        for c in range(r + 1, len(upper.columns)):
            val = upper.iloc[r, c]
            if val > 0.90:
                pairs.append(f"{upper.columns[r]} & {upper.columns[c]} ({val:.4f})")

    if pairs:
        for p in pairs[:5]:  # Print top 5
            print(p)
        if len(pairs) > 5:
            print(f"... and {len(pairs)-5} more.")
    else:
        print("None found.")
    print("")

    # 4. Feature Importance (Random Forest)
    print("Top 5 Features (Random Forest Importance):")
    X = df_analysis.drop("target", axis=1)
    y = df_analysis["target"]

    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, max_depth=5, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(indices))):
        print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")
    print("")

    # 5. Unstructured Relationship (Metadata vs Target)
    # Check if inc_angle distribution differs by class
    print("Metadata vs Target Relationship:")
    mean_angle_ship = df_analysis[df_analysis["target"] == 0]["inc_angle"].mean()
    mean_angle_ice = df_analysis[df_analysis["target"] == 1]["inc_angle"].mean()
    print(f"Mean inc_angle for Ships: {mean_angle_ship:.4f}")
    print(f"Mean inc_angle for Icebergs: {mean_angle_ice:.4f}")


def main():
    set_seed()

    # Load
    df_meta, samples = load_data()

    # Analyze
    analyze_target(df_meta)
    analyze_images(samples)
    analyze_tabular(df_meta)
    analyze_relationships(df_meta, samples)


if __name__ == "__main__":
    main()
