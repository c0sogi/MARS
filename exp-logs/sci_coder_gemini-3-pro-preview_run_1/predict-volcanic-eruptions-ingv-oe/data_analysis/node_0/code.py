import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Random Seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def main():
    # Paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_meta = pd.read_csv(METADATA_PATH)

    # ==========================================
    # 1. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    target = df_meta["time_to_eruption"]

    print(
        f"Distribution: Mean={target.mean():.4f}, Std={target.std():.4f}, "
        f"Min={target.min():.4f}, Max={target.max():.4f}"
    )

    skew = stats.skew(target)
    kurt = stats.kurtosis(target)
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    print("-" * 30)

    # ==========================================
    # 2. INPUT DATA ANALYSIS (TABULAR/SENSOR)
    # ==========================================
    print("INPUT DATA ANALYSIS (TABULAR/SENSOR)")

    # We treat the content of the CSV files as the "Tabular" data.
    # Since reading all files is too heavy, we sample 50 files for detailed distribution analysis.
    sample_size_detailed = 50
    sample_ids_detailed = (
        df_meta["segment_id"]
        .sample(n=min(sample_size_detailed, len(df_meta)), random_state=RANDOM_SEED)
        .values
    )

    # Accumulate data
    data_frames = []
    for seg_id in sample_ids_detailed:
        file_path = os.path.join(INPUT_DIR, "train", f"{seg_id}.csv")
        if os.path.exists(file_path):
            # Load as float32 to handle NaNs and reduce memory
            df_temp = pd.read_csv(file_path, dtype="float32")
            data_frames.append(df_temp)

    if not data_frames:
        print("No data files found for analysis.")
        return

    df_detailed = pd.concat(data_frames, ignore_index=True)

    # Numerical Analysis
    print("Numerical Statistics (Aggregated from Sample):")
    stats_df = df_detailed.describe().T[["mean", "std", "min", "max"]]
    # Calculate Outliers (IQR method)
    Q1 = df_detailed.quantile(0.25)
    Q3 = df_detailed.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (
        (df_detailed < (Q1 - 1.5 * IQR)) | (df_detailed > (Q3 + 1.5 * IQR))
    ).sum()

    # Format and print
    print(
        f"{'Sensor':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Outliers':<10}"
    )
    for col in stats_df.index:
        row = stats_df.loc[col]
        print(
            f"{col:<12} {row['mean']:<12.4f} {row['std']:<12.4f} {row['min']:<12.4f} {row['max']:<12.4f} {outliers[col]:<10}"
        )

    # Missing Values
    print("\nMissing Values:")
    missing = df_detailed.isnull().sum()
    missing_pct = (df_detailed.isnull().sum() / len(df_detailed)) * 100
    for col in df_detailed.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} ({missing_pct[col]:.4f}%)")
    if missing.sum() == 0:
        print("No missing values found in the sample.")

    # Categorical: None expected in sensor data
    print("\nCategorical Analysis:")
    print("No categorical columns detected (Sensor data is purely numerical).")
    print("-" * 30)

    # ==========================================
    # 3. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # For relationships, we need a dataset where each row is a segment.
    # We will extract features from a larger sample of files (e.g., 200) to correlate with the target.
    sample_size_features = 200
    sample_indices = df_meta.sample(
        n=min(sample_size_features, len(df_meta)), random_state=RANDOM_SEED
    ).index
    sample_meta = df_meta.loc[sample_indices].copy()

    feature_rows = []
    targets = []

    # Define features to extract per sensor
    sensors = [f"sensor_{i}" for i in range(1, 11)]

    for idx, row in sample_meta.iterrows():
        seg_id = row["segment_id"]
        target_val = row["time_to_eruption"]
        file_path = os.path.join(INPUT_DIR, "train", f"{seg_id}.csv")

        if os.path.exists(file_path):
            try:
                df_seg = pd.read_csv(file_path, dtype="float32")
                # Fill NaNs for feature extraction
                df_seg = df_seg.fillna(df_seg.mean())

                features = {}
                for sensor in sensors:
                    if sensor in df_seg.columns:
                        s_data = df_seg[sensor]
                        features[f"{sensor}_mean"] = s_data.mean()
                        features[f"{sensor}_std"] = s_data.std()
                        features[f"{sensor}_min"] = s_data.min()
                        features[f"{sensor}_max"] = s_data.max()
                        features[f"{sensor}_q50"] = s_data.median()

                feature_rows.append(features)
                targets.append(target_val)
            except Exception:
                continue

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets, name="time_to_eruption")

    if X.empty:
        print("Could not extract features for relationship analysis.")
        return

    # Structured Relationships
    # 1. Correlation
    print("Top 5 Feature Correlations with Target (Pearson):")
    # Add target to X temporarily for correlation
    X_corr = X.copy()
    X_corr["target"] = y
    correlations = X_corr.corr()["target"].drop("target")
    top_corr = correlations.abs().sort_values(ascending=False).head(5)
    for feat, corr_val in top_corr.items():
        # Retrieve original sign
        orig_val = correlations[feat]
        print(f"{feat}: {orig_val:.4f}")

    # 2. Importance (Random Forest)
    print("\nTop 5 Important Features (Random Forest):")
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_imp = importances.sort_values(ascending=False).head(5)
    for feat, imp in top_imp.items():
        print(f"{feat}: {imp:.4f}")

    # 3. Redundancy (Collinearity)
    print("\nRedundancy (Highly Collinear Pairs > 0.90):")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (column, index, upper.loc[index, column])
        for index in upper.index
        for column in upper.columns
        if upper.loc[index, column] > 0.90
    ]

    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    # Report top 5 redundant pairs to avoid clutter
    for i, (feat1, feat2, val) in enumerate(high_corr_pairs[:5]):
        print(f"{feat1} - {feat2}: {val:.4f}")
    if len(high_corr_pairs) > 5:
        print(f"... and {len(high_corr_pairs) - 5} more pairs.")
    elif len(high_corr_pairs) == 0:
        print("No highly collinear pairs found.")

    # Unstructured / Metadata Relationships
    # Here we check if specific metadata properties correlate with target.
    # Since we don't have extra metadata (like 'sensor_type' or 'location'),
    # we can check if 'Signal Length' (file size/rows) varies.
    # Note: The problem description implies fixed 10 minutes, so rows should be constant ~60k.
    # We'll verify this assumption.

    print("\nMetadata-Target Relationship (Signal Length):")
    # Check if row count correlates with target
    # We need to re-read row counts for the feature sample
    lengths = []
    for idx, row in sample_meta.iterrows():
        seg_id = row["segment_id"]
        file_path = os.path.join(INPUT_DIR, "train", f"{seg_id}.csv")
        if os.path.exists(file_path):
            try:
                # Just read header to skip, or read quickly
                with open(file_path) as f:
                    row_count = sum(1 for _ in f) - 1  # minus header
                lengths.append(row_count)
            except:
                lengths.append(np.nan)
        else:
            lengths.append(np.nan)

    length_corr = pd.Series(lengths).corr(pd.Series(targets))
    print(f"Correlation between Signal Length (rows) and Target: {length_corr:.4f}")
    print(f"Average Signal Length: {np.nanmean(lengths):.4f} rows")


if __name__ == "__main__":
    main()
