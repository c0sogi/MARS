import os
import gc
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# =============================================================================
# Configuration & Setup
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def format_float(val):
    return f"{val:.4f}"


def print_section(title):
    print(f"\n{title.upper()}")
    print("=" * len(title))


def main():
    # =========================================================================
    # 1. Data Loading (Metadata & Geometry)
    # =========================================================================
    print_section("1. Data Loading")

    # Load Sensor Geometry
    # The sensor_geometry.csv has sensor_id as index usually, let's check
    geo_df = pd.read_csv(SENSOR_GEO_PATH)
    print(f"Sensor Geometry Loaded: {geo_df.shape[0]} sensors.")

    # Load Train Metadata
    # We load the whole file to analyze targets, but will sample for feature analysis
    print(f"Loading training metadata from {TRAIN_META_PATH}...")
    meta_df = pd.read_parquet(TRAIN_META_PATH)
    print(f"Training Metadata Loaded: {len(meta_df)} events.")

    # =========================================================================
    # 2. Target Variable Analysis
    # =========================================================================
    print_section("2. Target Variable Analysis")

    targets = ["azimuth", "zenith"]

    for target in targets:
        data = meta_df[target].values

        # Basic Stats
        mu = np.mean(data)
        sigma = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # Normality Checks
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        print(f"Target: {target}")
        print(f"  Range: [{format_float(min_val)}, {format_float(max_val)}]")
        print(f"  Mean:  {format_float(mu)}")
        print(f"  Std:   {format_float(sigma)}")
        print(f"  Skewness: {format_float(skew)} (Normal=0)")
        print(f"  Kurtosis: {format_float(kurt)} (Normal=0)")

        if abs(skew) < 0.5:
            print("  Distribution: Approximately Symmetric")
        else:
            print("  Distribution: Skewed")
        print("-" * 30)

    # =========================================================================
    # 3. Input Data Analysis (Tabular/Signal Aggregation)
    # =========================================================================
    print_section("3. Input Data Analysis (Pulse Aggregation)")

    # Strategy: Since data is split into batches, we cannot process 95M events for detailed feature stats.
    # We will sample 3 batches to perform detailed input analysis and feature engineering.

    sample_batch_ids = meta_df["batch_id"].unique()[:3]  # Take first 3 batches
    print(f"Sampling data from batches: {sample_batch_ids}")

    pulse_data_list = []

    for bid in sample_batch_ids:
        batch_path = os.path.join(INPUT_DIR, f"train/batch_{bid}.parquet")
        if os.path.exists(batch_path):
            batch_df = pd.read_parquet(batch_path).reset_index()
            pulse_data_list.append(batch_df)

    raw_pulses = pd.concat(pulse_data_list, ignore_index=True)

    # Merge geometry
    # sensor_geometry.csv usually has 'sensor_id', 'x', 'y', 'z'
    # raw_pulses has 'sensor_id'
    raw_pulses = raw_pulses.merge(geo_df, on="sensor_id", how="left")

    print(
        f"Analyzed Sample Size: {len(raw_pulses)} pulses from {len(raw_pulses['event_id'].unique())} events."
    )

    # --- Raw Pulse Statistics (Signal Level) ---
    print("\n[Signal Statistics - Raw Pulses]")
    signal_cols = ["charge", "time", "x", "y", "z"]

    for col in signal_cols:
        col_data = raw_pulses[col].dropna()
        mean_val = col_data.mean()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()

        # Outliers (IQR method)
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {format_float(mean_val)}, Std: {format_float(std_val)}")
        print(f"  Min:  {format_float(min_val)}, Max: {format_float(max_val)}")
        print(
            f"  Outliers (IQR): {outliers} ({format_float(outliers/len(col_data)*100)}%)"
        )
        print(f"  Missing: {raw_pulses[col].isna().sum()}")

    # --- Feature Engineering for Event Level Analysis ---
    print("\n[Aggregating Pulses to Events...]")

    # We calculate simple physical features per event
    # 1. n_pulses: Count
    # 2. total_charge: Sum of charge
    # 3. time_duration: max(time) - min(time)
    # 4. mean_x/y/z: Charge-weighted center of gravity

    # Helper for weighted average
    def weighted_avg_and_std(values, weights):
        average = np.average(values, weights=weights)
        return average

    # Group by event_id
    # Note: This can be slow, so we use pandas optimized operations

    # Pre-calculate weighted coordinates
    raw_pulses["wx"] = raw_pulses["x"] * raw_pulses["charge"]
    raw_pulses["wy"] = raw_pulses["y"] * raw_pulses["charge"]
    raw_pulses["wz"] = raw_pulses["z"] * raw_pulses["charge"]

    aggs = {
        "charge": ["sum", "mean", "std"],
        "time": ["min", "max", "count"],
        "wx": ["sum"],
        "wy": ["sum"],
        "wz": ["sum"],
        "auxiliary": ["mean"],  # ratio of aux pulses
    }

    event_features = raw_pulses.groupby("event_id").agg(aggs)
    event_features.columns = [
        "_".join(col).strip() for col in event_features.columns.values
    ]

    # Post-processing
    event_features["n_pulses"] = event_features["time_count"]
    event_features["total_charge"] = event_features["charge_sum"]
    event_features["time_duration"] = (
        event_features["time_max"] - event_features["time_min"]
    )

    # Center of Gravity (Charge Weighted)
    # Avoid division by zero if total charge is 0 (unlikely but safe)
    mask = event_features["total_charge"] > 0
    event_features.loc[mask, "center_x"] = (
        event_features.loc[mask, "wx_sum"] / event_features.loc[mask, "total_charge"]
    )
    event_features.loc[mask, "center_y"] = (
        event_features.loc[mask, "wy_sum"] / event_features.loc[mask, "total_charge"]
    )
    event_features.loc[mask, "center_z"] = (
        event_features.loc[mask, "wz_sum"] / event_features.loc[mask, "total_charge"]
    )

    # Fill NaNs for events with 0 charge (if any)
    event_features.fillna(0, inplace=True)

    # Select final features for analysis
    feature_cols = [
        "n_pulses",
        "total_charge",
        "charge_mean",
        "charge_std",
        "time_duration",
        "auxiliary_mean",
        "center_x",
        "center_y",
        "center_z",
    ]
    df_events = event_features[feature_cols].copy()

    # Merge with targets
    # meta_df has all events, we filter for the ones we sampled
    df_events = df_events.merge(
        meta_df[["event_id", "azimuth", "zenith"]], on="event_id", how="inner"
    )

    print(f"Aggregated Event Dataset Shape: {df_events.shape}")

    # Tabular Analysis of Aggregated Features
    print("\n[Event-Level Feature Statistics]")
    for col in feature_cols:
        col_data = df_events[col]
        print(f"Feature: {col}")
        print(f"  Mean: {format_float(col_data.mean())}")
        print(f"  Std:  {format_float(col_data.std())}")
        print(f"  Min:  {format_float(col_data.min())}")
        print(f"  Max:  {format_float(col_data.max())}")

    # =========================================================================
    # 4. Feature/Signal Relationships
    # =========================================================================
    print_section("4. Feature/Signal Relationships")

    # 4.1 Correlations
    print("[Correlation Analysis (Pearson)]")
    corr_matrix = df_events[feature_cols + targets].corr()

    print("Correlation with Azimuth:")
    print(
        corr_matrix["azimuth"]
        .drop(targets)
        .sort_values(ascending=False)
        .apply(format_float)
    )

    print("\nCorrelation with Zenith:")
    print(
        corr_matrix["zenith"]
        .drop(targets)
        .sort_values(ascending=False)
        .apply(format_float)
    )

    # 4.2 Redundancy (Collinearity)
    print("\n[Redundancy Check (Correlation > 0.90)]")
    high_corr_pairs = []
    features_only = df_events[feature_cols].corr()
    for i in range(len(features_only.columns)):
        for j in range(i + 1, len(features_only.columns)):
            if abs(features_only.iloc[i, j]) > 0.90:
                high_corr_pairs.append(
                    (
                        features_only.columns[i],
                        features_only.columns[j],
                        features_only.iloc[i, j],
                    )
                )

    if high_corr_pairs:
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} <-> {f2}: {format_float(val)}")
    else:
        print("  No highly collinear pairs found.")

    # 4.3 Feature Importance (Random Forest)
    print("\n[Feature Importance (Random Forest)]")
    # We train two separate lightweight models for Azimuth and Zenith

    # Sample for training to keep it fast
    train_sample_size = min(50000, len(df_events))
    df_train = df_events.sample(n=train_sample_size, random_state=SEED)

    X = df_train[feature_cols].fillna(0)

    for target in targets:
        y = df_train[target]

        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=feature_cols)
        print(f"\nTop 5 Features for {target}:")
        print(importances.sort_values(ascending=False).head(5).apply(format_float))

    # =========================================================================
    # 5. Metadata Relationships
    # =========================================================================
    print_section("5. Metadata Relationships")

    # Check if 'n_pulses' (proxy for energy/size) correlates with targets
    # We already did this in correlation, but let's explicitly interpret

    corr_n_pulses_zenith = df_events["n_pulses"].corr(df_events["zenith"])
    print(f"Relationship: Event Size (n_pulses) vs Zenith")
    print(f"  Correlation: {format_float(corr_n_pulses_zenith)}")
    if abs(corr_n_pulses_zenith) > 0.1:
        print("  Observation: Larger events show some dependency with Zenith angle.")
    else:
        print("  Observation: Event size is largely independent of Zenith angle.")

    # Clean up
    del raw_pulses, df_events, meta_df
    gc.collect()


if __name__ == "__main__":
    main()
