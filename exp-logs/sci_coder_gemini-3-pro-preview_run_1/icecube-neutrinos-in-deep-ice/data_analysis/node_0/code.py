import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # ==========================================
    # 1. Load Metadata & Targets
    # ==========================================
    # Load full training metadata for target analysis
    meta_path = os.path.join(METADATA_DIR, "train_metadata.parquet")
    df_meta = pd.read_parquet(meta_path)

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("2. TARGET VARIABLE ANALYSIS")

    targets = ["azimuth", "zenith"]

    for target in targets:
        data = df_meta[target]
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        skew = data.skew()
        kurt = data.kurtosis()

        print(f"Variable: {target}")
        print(
            f"  Distribution: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}"
        )
        print(f"  Normality: Skewness={skew:.4f}, Kurtosis={kurt:.4f}")

    # ==========================================
    # 3. Input Data Analysis
    # ==========================================
    print("\n3. INPUT DATA ANALYSIS")

    # Load Geometry for later use
    geo_path = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    df_geo = pd.read_csv(geo_path)
    sensor_map = df_geo.set_index("sensor_id")[["x", "y", "z"]].to_dict("index")

    # Sample Data Strategy:
    # We cannot load all TBs of data. We will sample 3 random batches.
    unique_batches = df_meta["batch_id"].unique()
    sampled_batches = np.random.choice(unique_batches, size=3, replace=False)

    # Containers for analysis
    pulse_stats = {"charge": [], "time": [], "auxiliary": [], "sensor_id": []}
    event_lengths = []

    # Container for Section 4 (Feature Relationships)
    # We will build a dataframe of aggregated features
    agg_features_list = []

    for batch_id in sampled_batches:
        # Load batch file
        batch_rel_path = f"train/batch_{batch_id}.parquet"
        batch_full_path = os.path.join(INPUT_DIR, batch_rel_path)

        if not os.path.exists(batch_full_path):
            continue

        df_batch = pd.read_parquet(batch_full_path)

        # --- Collect Pulse Level Stats (Subsampling for speed if needed, but batches are manageable) ---
        # We'll take a 10% sample of pulses for distribution stats to save memory/time
        df_sample_pulses = df_batch.sample(frac=0.1, random_state=42)

        pulse_stats["charge"].extend(df_sample_pulses["charge"].tolist())
        pulse_stats["time"].extend(df_sample_pulses["time"].tolist())
        pulse_stats["auxiliary"].extend(df_sample_pulses["auxiliary"].tolist())
        pulse_stats["sensor_id"].extend(df_sample_pulses["sensor_id"].tolist())

        # --- Aggregate for Event Level Analysis & Section 4 ---
        # Group by event_id
        # We need to map sensor_id to x,y,z first for spatial features
        # To do this efficiently, we merge geometry
        df_batch_geo = df_batch.merge(df_geo, on="sensor_id", how="left")

        # Aggregations
        # 1. Count pulses (event length)
        # 2. Sum charge
        # 3. Time duration
        # 4. Weighted Centroids

        # Pre-calculate weighted positions
        df_batch_geo["wx"] = df_batch_geo["x"] * df_batch_geo["charge"]
        df_batch_geo["wy"] = df_batch_geo["y"] * df_batch_geo["charge"]
        df_batch_geo["wz"] = df_batch_geo["z"] * df_batch_geo["charge"]

        grp = df_batch_geo.groupby("event_id")

        aggs = grp.agg(
            {
                "charge": ["count", "sum", "mean"],
                "time": ["min", "max"],
                "auxiliary": "mean",  # ratio
                "wx": "sum",
                "wy": "sum",
                "wz": "sum",
            }
        )

        # Flatten columns
        aggs.columns = [
            "n_pulses",
            "total_charge",
            "mean_charge",
            "min_time",
            "max_time",
            "aux_ratio",
            "sum_wx",
            "sum_wy",
            "sum_wz",
        ]
        aggs["duration"] = aggs["max_time"] - aggs["min_time"]
        aggs["center_x"] = aggs["sum_wx"] / aggs["total_charge"]
        aggs["center_y"] = aggs["sum_wy"] / aggs["total_charge"]
        aggs["center_z"] = aggs["sum_wz"] / aggs["total_charge"]

        # Handle potential division by zero if total_charge is 0 (unlikely but possible)
        aggs = aggs.fillna(0)

        # Store event lengths for Section 3
        event_lengths.extend(aggs["n_pulses"].tolist())

        # Join with targets for Section 4
        # Get targets for these events
        batch_meta = df_meta[df_meta["batch_id"] == batch_id].set_index("event_id")

        # Inner join to ensure alignment
        batch_features = aggs.join(batch_meta[["azimuth", "zenith"]], how="inner")
        agg_features_list.append(batch_features)

    # --- Process Section 3 Stats ---

    # Numerical: Charge
    charges = np.array(pulse_stats["charge"])
    q1_c, q3_c = np.percentile(charges, [25, 75])
    iqr_c = q3_c - q1_c
    lower_c = q1_c - 1.5 * iqr_c
    upper_c = q3_c + 1.5 * iqr_c
    outliers_c = ((charges < lower_c) | (charges > upper_c)).sum()

    print("Numerical Columns:")
    print(f"  Column: charge")
    print(
        f"    Stats: Mean={np.mean(charges):.4f}, Std={np.std(charges):.4f}, Min={np.min(charges):.4f}, Max={np.max(charges):.4f}"
    )
    print(f"    Outliers (IQR): {outliers_c} ({outliers_c/len(charges)*100:.2f}%)")

    # Numerical: Time
    times = np.array(pulse_stats["time"])
    # Time is relative, so global stats are less meaningful for outliers, but range matters
    print(f"  Column: time")
    print(
        f"    Stats: Mean={np.mean(times):.4f}, Std={np.std(times):.4f}, Min={np.min(times):.4f}, Max={np.max(times):.4f}"
    )

    # Categorical: Sensor ID
    unique_sensors = len(np.unique(pulse_stats["sensor_id"]))
    print("Categorical Columns:")
    print(f"  Column: sensor_id")
    print(f"    Cardinality: {unique_sensors}")
    if unique_sensors > 50:
        print(f"    Flag: High cardinality (>50 categories).")

    # Categorical/Boolean: Auxiliary
    aux_counts = pd.Series(pulse_stats["auxiliary"]).value_counts()
    aux_true_ratio = aux_counts.get(True, 0) / len(pulse_stats["auxiliary"])
    print(f"  Column: auxiliary")
    print(f"    Class Balance: True={aux_true_ratio:.4f}, False={1-aux_true_ratio:.4f}")

    # Missing Values
    # Parquet load usually handles this, but we check our sample
    # Since we loaded into lists, we check NaNs in the lists (converted to array)
    nan_charge = np.isnan(charges).sum()
    print("Missing Values:")
    print(f"  Column: charge, NaNs: {nan_charge} ({nan_charge/len(charges)*100:.4f}%)")

    # Sequence Length Analysis (Event based)
    lens = np.array(event_lengths)
    print("Sequence Lengths (Pulses per Event):")
    print(
        f"  Stats: Mean={np.mean(lens):.4f}, Std={np.std(lens):.4f}, Min={np.min(lens):.4f}, Max={np.max(lens):.4f}"
    )

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\n4. FEATURE/SIGNAL RELATIONSHIPS")

    # Combine all processed batches
    df_analysis = pd.concat(agg_features_list, axis=0)

    # Define features and targets
    feature_cols = [
        "n_pulses",
        "total_charge",
        "mean_charge",
        "duration",
        "aux_ratio",
        "center_x",
        "center_y",
        "center_z",
    ]
    target_cols = ["azimuth", "zenith"]

    # 4.1 Correlations
    print("Structured Relationships:")
    corr_matrix = df_analysis[feature_cols + target_cols].corr(method="pearson")

    print("  Correlations with Targets:")
    for t in target_cols:
        print(f"    Target: {t}")
        for f in feature_cols:
            c = corr_matrix.loc[f, t]
            print(f"      {f}: {c:.4f}")

    # 4.2 Importance (Random Forest)
    # We'll use a subset for training to keep it lightweight
    if len(df_analysis) > 50000:
        df_rf = df_analysis.sample(50000, random_state=42)
    else:
        df_rf = df_analysis

    X = df_rf[feature_cols].fillna(0)  # Simple imputation
    y = df_rf[target_cols]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    rf.fit(X, y)

    # Feature importance is averaged over trees. Since it's multi-output, sklearn averages over outputs for feature_importances_
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Feature Importance (Top 5):")
    for i in range(min(5, len(feature_cols))):
        idx = indices[i]
        print(f"    {feature_cols[idx]}: {importances[idx]:.4f}")

    # 4.3 Redundancy
    print("  Redundancy (Collinear Pairs > 0.90):")
    found_redundancy = False
    # Check feature-feature correlation
    feat_corr = df_analysis[feature_cols].corr().abs()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if feat_corr.iloc[i, j] > 0.90:
                print(
                    f"    {feature_cols[i]} - {feature_cols[j]}: {feat_corr.iloc[i, j]:.4f}"
                )
                found_redundancy = True
    if not found_redundancy:
        print("    None found.")

    # Unstructured / Meta-Feature Relationships
    # Here we analyze if "Event Size" (n_pulses) correlates with targets
    print("Unstructured Relationships:")
    # We already have n_pulses in the correlation matrix, but let's interpret it conceptually as requested
    corr_len_azi = corr_matrix.loc["n_pulses", "azimuth"]
    corr_len_zen = corr_matrix.loc["n_pulses", "zenith"]
    print(
        f"  Event Length vs Targets: Azimuth Corr={corr_len_azi:.4f}, Zenith Corr={corr_len_zen:.4f}"
    )

    # Check if total charge (signal strength) correlates
    corr_chg_azi = corr_matrix.loc["total_charge", "azimuth"]
    corr_chg_zen = corr_matrix.loc["total_charge", "zenith"]
    print(
        f"  Signal Strength (Charge) vs Targets: Azimuth Corr={corr_chg_azi:.4f}, Zenith Corr={corr_chg_zen:.4f}"
    )


if __name__ == "__main__":
    main()
