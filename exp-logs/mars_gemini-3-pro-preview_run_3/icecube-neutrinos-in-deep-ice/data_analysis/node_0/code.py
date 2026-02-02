import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
METADATA_PATH = "./metadata/train_metadata.parquet"
SAMPLE_BATCH_PATH = "./input/train/batch_1.parquet"
SENSOR_GEOMETRY_PATH = "./input/sensor_geometry.csv"
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)


def print_section_header(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_targets():
    print_section_header("Target Variable Analysis")

    # Load only target columns to save memory/time
    # The metadata file contains all training events
    try:
        df_meta = pd.read_parquet(METADATA_PATH, columns=["azimuth", "zenith"])
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    targets = ["azimuth", "zenith"]

    for target in targets:
        data = df_meta[target]
        print(f"\nVariable: {target}")

        # Distribution Stats
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std:  {data.std():.4f}")
        print(f"  Min:  {data.min():.4f}")
        print(f"  Max:  {data.max():.4f}")

        # Normality Checks
        target_skew = skew(data)
        target_kurt = kurtosis(data)
        print(f"  Skewness: {target_skew:.4f}")
        print(f"  Kurtosis: {target_kurt:.4f}")

        # Interpretation
        if abs(target_skew) < 0.5:
            dist_shape = "Approximately Symmetric"
        else:
            dist_shape = "Skewed"
        print(f"  Distribution Shape: {dist_shape}")


def analyze_inputs():
    print_section_header("Input Data Analysis (Tabular/Pulse Level)")

    # Load a single batch to analyze input feature statistics
    # This represents the raw data modality (Tabular sequence of pulses)
    if not os.path.exists(SAMPLE_BATCH_PATH):
        print(f"Sample batch file not found at {SAMPLE_BATCH_PATH}")
        return

    df_batch = pd.read_parquet(SAMPLE_BATCH_PATH)

    # 1. Missing Values
    print("\n--- Missing Values ---")
    missing = df_batch.isna().sum()
    total_rows = len(df_batch)
    for col in df_batch.columns:
        cnt = missing[col]
        pct = (cnt / total_rows) * 100
        print(f"  {col}: {cnt} missing ({pct:.4f}%)")

    # 2. Numerical Features Analysis
    num_cols = ["time", "charge"]
    print("\n--- Numerical Features ---")

    for col in num_cols:
        data = df_batch[col]
        print(f"\n  Feature: {col}")
        print(f"    Mean: {data.mean():.4f}")
        print(f"    Std:  {data.std():.4f}")
        print(f"    Min:  {data.min():.4f}")
        print(f"    Max:  {data.max():.4f}")

        # Outlier Detection (IQR)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        print(f"    Outliers (IQR method): {outliers} ({outliers/total_rows*100:.2f}%)")

    # 3. Categorical/Discrete Features Analysis
    print("\n--- Categorical/Discrete Features ---")

    # Sensor ID
    n_sensors = df_batch["sensor_id"].nunique()
    print(f"\n  Feature: sensor_id")
    print(f"    Cardinality: {n_sensors} unique sensors used in this batch")
    # Check for rare sensors is less relevant here as all are valid physical sensors,
    # but we can check coverage.
    print(
        f"    Coverage: {n_sensors/5160*100:.2f}% of total IceCube sensors active in this batch"
    )

    # Auxiliary
    print(f"\n  Feature: auxiliary")
    aux_counts = df_batch["auxiliary"].value_counts()
    for val, count in aux_counts.items():
        print(f"    Value {val}: {count} ({count/total_rows*100:.2f}%)")


def analyze_relationships():
    print_section_header("Feature/Signal Relationships")

    # To analyze relationships, we need to aggregate the pulse-level data to the event-level
    # and merge with the targets.

    # 1. Load Data
    df_batch = pd.read_parquet(SAMPLE_BATCH_PATH)

    # Load metadata for this specific batch
    # We infer batch ID from the filename (batch_1 -> 1)
    batch_id = 1
    df_meta = pd.read_parquet(METADATA_PATH)
    df_meta_batch = df_meta[df_meta["batch_id"] == batch_id].copy()

    # 2. Feature Engineering (Aggregation)
    print("Aggregating pulse data to event level...")

    # Basic aggregations
    agg_funcs = {
        "charge": ["sum", "mean", "max", "count"],  # count represents n_pulses
        "time": ["min", "max", "std"],
        "auxiliary": ["mean"],  # ratio of auxiliary pulses
    }

    df_agg = df_batch.groupby("event_id").agg(agg_funcs)

    # Flatten MultiIndex columns
    df_agg.columns = [f"{col}_{stat}" for col, stat in df_agg.columns]
    df_agg = df_agg.reset_index()

    # Add derived features
    df_agg["event_duration"] = df_agg["time_max"] - df_agg["time_min"]

    # 3. Merge with Targets
    df_merged = pd.merge(
        df_agg,
        df_meta_batch[["event_id", "azimuth", "zenith"]],
        on="event_id",
        how="inner",
    )

    if df_merged.empty:
        print("Error: No overlapping events found between batch and metadata.")
        return

    # 4. Correlation Analysis
    print("\n--- Correlation Analysis (Pearson) ---")
    # Select numerical features for correlation
    features = [
        c for c in df_merged.columns if c not in ["event_id", "azimuth", "zenith"]
    ]

    # Check correlation with Zenith (physically more linked to energy/track length than Azimuth)
    correlations = df_merged[features + ["zenith"]].corr()["zenith"].drop("zenith")

    print("Top 5 Features correlated with Zenith:")
    print(
        correlations.abs()
        .sort_values(ascending=False)
        .head(5)
        .to_string(float_format="{:.4f}".format)
    )

    # Check Collinearity
    print("\n--- Redundancy Check (Collinearity > 0.90) ---")
    corr_matrix = df_merged[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        print(f"Highly collinear features detected: {to_drop}")
    else:
        print("No highly collinear features detected among aggregates.")

    # 5. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    print("Training lightweight Random Forest to predict Zenith...")

    X = df_merged[features].fillna(0)  # Handle NaNs from std of single-pulse events
    y = df_merged["zenith"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=7, n_jobs=-1, random_state=SEED, verbose=0
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features)
    print("\nTop 5 Important Features for Zenith Prediction:")
    print(
        importances.sort_values(ascending=False)
        .head(5)
        .to_string(float_format="{:.4f}".format)
    )

    # 6. Meta-Feature Insight
    print("\n--- Meta-Feature Insight ---")
    # Does the number of pulses correlate with the target?
    corr_pulses = df_merged["charge_count"].corr(df_merged["zenith"])
    print(f"Correlation between Pulse Count and Zenith: {corr_pulses:.4f}")
    print(
        "Interpretation: A higher magnitude correlation implies that event 'size' (number of pulses) contains information about the vertical angle."
    )


def main():
    print("Starting Exploratory Data Analysis...")

    analyze_targets()
    analyze_inputs()
    analyze_relationships()

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
