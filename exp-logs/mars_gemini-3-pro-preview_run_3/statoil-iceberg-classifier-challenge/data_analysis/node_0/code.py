import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score


def set_seed(seed=42):
    np.random.seed(seed)
    # sklearn uses numpy random state usually, but we set it explicitly where needed


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    target_col = "is_iceberg"

    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total}")
    print("Class Distribution:")
    for label, count in counts.items():
        ratio = count / total
        label_name = "Iceberg (1)" if label == 1 else "Ship (0)"
        print(f"  {label_name}: {count} ({ratio:.4f})")

    # Imbalance check
    ratio_0 = counts.get(0, 0) / total
    ratio_1 = counts.get(1, 0) / total
    print(f"Class Balance Ratio (0:1): {ratio_0:.4f} : {ratio_1:.4f}")
    print("-" * 30)


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")

    # Convert lists to numpy arrays for vectorized ops
    # Band 1
    b1_matrix = np.array(df["band_1"].tolist())
    # Band 2
    b2_matrix = np.array(df["band_2"].tolist())

    n_samples, n_pixels = b1_matrix.shape

    # Assuming square images based on description (75x75 = 5625)
    width = int(np.sqrt(n_pixels))
    height = width

    print(f"Image Dimensions: {width}x{height} (Flattened length: {n_pixels})")
    print(f"Channel Count: 2 (Band 1, Band 2)")

    # Check consistency
    if n_pixels != 75 * 75:
        print(f"WARNING: Pixel count {n_pixels} does not match expected 75x75=5625")

    # Global Pixel Stats
    print("\nGlobal Pixel Statistics (Band 1 - HH):")
    print(f"  Mean: {np.mean(b1_matrix):.4f}")
    print(f"  Std : {np.std(b1_matrix):.4f}")
    print(f"  Min : {np.min(b1_matrix):.4f}")
    print(f"  Max : {np.max(b1_matrix):.4f}")

    print("\nGlobal Pixel Statistics (Band 2 - HV):")
    print(f"  Mean: {np.mean(b2_matrix):.4f}")
    print(f"  Std : {np.std(b2_matrix):.4f}")
    print(f"  Min : {np.min(b2_matrix):.4f}")
    print(f"  Max : {np.max(b2_matrix):.4f}")

    return b1_matrix, b2_matrix


def analyze_tabular(df):
    print("-" * 30)
    print("INPUT DATA ANALYSIS (TABULAR)")

    col = "inc_angle"
    series = df[col]

    # Missing values
    n_missing = series.isna().sum()
    p_missing = (n_missing / len(df)) * 100
    print(f"Feature: {col}")
    print(f"  Missing Values: {n_missing} ({p_missing:.4f}%)")

    # Stats on non-missing
    valid_series = series.dropna()
    if len(valid_series) > 0:
        desc = valid_series.describe()
        print(f"  Mean: {desc['mean']:.4f}")
        print(f"  Std : {desc['std']:.4f}")
        print(f"  Min : {desc['min']:.4f}")
        print(f"  Max : {desc['max']:.4f}")

        # Outliers (IQR)
        Q1 = valid_series.quantile(0.25)
        Q3 = valid_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = valid_series[
            (valid_series < lower_bound) | (valid_series > upper_bound)
        ]
        print(f"  Outlier Count (IQR method): {len(outliers)}")
    else:
        print("  No valid numerical data available.")
    print("-" * 30)


def analyze_relationships(df, b1_matrix, b2_matrix):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Feature Engineering for Analysis
    # Create summary stats per image to correlate with target
    features = pd.DataFrame()
    features["b1_mean"] = np.mean(b1_matrix, axis=1)
    features["b1_std"] = np.std(b1_matrix, axis=1)
    features["b1_min"] = np.min(b1_matrix, axis=1)
    features["b1_max"] = np.max(b1_matrix, axis=1)

    features["b2_mean"] = np.mean(b2_matrix, axis=1)
    features["b2_std"] = np.std(b2_matrix, axis=1)
    features["b2_min"] = np.min(b2_matrix, axis=1)
    features["b2_max"] = np.max(b2_matrix, axis=1)

    features["inc_angle"] = df["inc_angle"].values
    target = df["is_iceberg"].values

    # 2. Correlations
    # Handle missing inc_angle for correlation by dropping just for that calculation
    corr_df = features.copy()
    corr_df["target"] = target
    corr_matrix = corr_df.corr(method="pearson")

    print("Top Correlations with Target (is_iceberg):")
    target_corr = (
        corr_matrix["target"].drop("target").abs().sort_values(ascending=False)
    )
    for name, val in target_corr.head(5).items():
        print(f"  {name}: {val:.4f}")

    print("\nCollinear Features (Correlation > 0.90):")
    # Check lower triangle to avoid duplicates
    cols = features.columns
    found_collinear = False
    for i in range(len(cols)):
        for j in range(i):
            c1 = cols[i]
            c2 = cols[j]
            # Use a temporary df dropping NaNs for this pair
            pair_corr = features[[c1, c2]].corr().iloc[0, 1]
            if abs(pair_corr) > 0.90:
                print(f"  {c1} & {c2}: {pair_corr:.4f}")
                found_collinear = True
    if not found_collinear:
        print("  None found.")

    # 3. Meta-Feature Relationship
    # Does inc_angle distribution differ by class?
    print("\nMeta-Feature Analysis (inc_angle vs Target):")
    mask_iceberg = target == 1
    mask_ship = target == 0

    inc_iceberg = features.loc[mask_iceberg, "inc_angle"].dropna()
    inc_ship = features.loc[mask_ship, "inc_angle"].dropna()

    print(f"  Avg inc_angle (Iceberg): {inc_iceberg.mean():.4f}")
    print(f"  Avg inc_angle (Ship)   : {inc_ship.mean():.4f}")

    # 4. Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")

    # Impute missing inc_angle
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(features)
    y = target

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features:")
    for i in range(5):
        idx = indices[i]
        print(f"  {features.columns[idx]}: {importances[idx]:.4f}")


def main():
    set_seed(42)

    # Load Metadata
    train_meta_path = "./metadata/train.csv"
    if not os.path.exists(train_meta_path):
        print("Metadata not found.")
        return

    df_meta = pd.read_csv(train_meta_path)
    train_ids = set(df_meta["id"].unique())

    # Load Raw Data
    raw_path = "./input/train.json"
    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    # Filter for training set only
    train_data = [item for item in raw_data if item["id"] in train_ids]
    df = pd.DataFrame(train_data)

    # Preprocessing: Convert inc_angle to numeric (coercing 'na' to NaN)
    df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")

    # Execution
    analyze_target(df)
    b1, b2 = analyze_images(df)
    analyze_tabular(df)
    analyze_relationships(df, b1, b2)


if __name__ == "__main__":
    main()
