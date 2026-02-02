import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42

# Set seeds
np.random.seed(SEED)


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


def analyze_numerical(df, columns):
    print(
        f"{'Column':<35} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Skew':<8} | {'NaN %':<7}"
    )
    print("-" * 110)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        skew_val = skew(series.dropna())
        nan_pct = (series.isna().sum() / len(series)) * 100

        print(
            f"{col:<35} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f} | {skew_val:<8.4f} | {nan_pct:<7.2f}"
        )


def analyze_categorical(df, columns):
    print(
        f"{'Column':<30} | {'Unique':<8} | {'Top Label':<20} | {'Freq %':<8} | {'Rare Labels (<1%)'}"
    )
    print("-" * 100)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        n_unique = series.nunique()
        value_counts = series.value_counts(normalize=True)
        top_label = value_counts.index[0]
        top_freq = value_counts.iloc[0] * 100
        rare_count = (value_counts < 0.01).sum()

        print(
            f"{col:<30} | {n_unique:<8} | {str(top_label)[:20]:<20} | {top_freq:<8.2f} | {rare_count}"
        )


def run_eda():
    print_header("1. Data Loading & Integrity")

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Metadata (which acts as Ground Truth Aggregation)
    meta_df = pd.read_csv(METADATA_PATH)
    print(f"Loaded Metadata Rows: {len(meta_df)}")
    print(f"Unique Drives: {meta_df['drive_id'].nunique()}")
    print(f"Unique Phones: {meta_df['phone_name'].nunique()}")

    # ---------------------------------------------------------
    print_header("2. Target Variable Analysis (Ground Truth)")
    # Targets: LatitudeDegrees, LongitudeDegrees
    # Auxiliary GT: AltitudeMeters, SpeedMps, AccuracyMeters (Need to load from actual GT files for these if not in metadata)

    # Metadata only has Lat/Lon. Let's load full GT for a sample to get Speed/Altitude/Accuracy
    sample_drives = (
        meta_df["drive_id"].drop_duplicates().sample(3, random_state=SEED).tolist()
    )
    print(f"Sampling full Ground Truth from drives: {sample_drives}")

    gt_dfs = []
    for drive in sample_drives:
        # Get one row per drive to find the path
        drive_row = meta_df[meta_df["drive_id"] == drive].iloc[0]
        gt_rel_path = drive_row["gt_path"]
        gt_abs_path = os.path.join(INPUT_DIR, gt_rel_path)
        if os.path.exists(gt_abs_path):
            df = pd.read_csv(gt_abs_path)
            gt_dfs.append(df)

    if gt_dfs:
        full_gt_sample = pd.concat(gt_dfs, ignore_index=True)
        target_cols = [
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
            "SpeedMps",
            "AccuracyMeters",
            "BearingDegrees",
        ]
        analyze_numerical(full_gt_sample, target_cols)
    else:
        print("Could not load full ground truth files.")

    # ---------------------------------------------------------
    print_header("3. Input Data Analysis (GNSS & IMU)")

    # We will use the same sample drives to load GNSS and IMU data
    gnss_dfs = []
    imu_dfs = []

    # Limit rows to prevent OOM
    MAX_ROWS = 50000

    for drive in sample_drives:
        # Get paths for this drive (take first phone found)
        drive_subset = meta_df[meta_df["drive_id"] == drive]
        if drive_subset.empty:
            continue

        row = drive_subset.iloc[0]

        # GNSS
        gnss_path = os.path.join(INPUT_DIR, row["gnss_path"])
        if os.path.exists(gnss_path):
            try:
                g_df = pd.read_csv(gnss_path, nrows=MAX_ROWS)
                gnss_dfs.append(g_df)
            except Exception as e:
                print(f"Failed to read GNSS: {e}")

        # IMU
        imu_path = os.path.join(INPUT_DIR, row["imu_path"])
        if os.path.exists(imu_path):
            try:
                i_df = pd.read_csv(imu_path, nrows=MAX_ROWS)
                imu_dfs.append(i_df)
            except Exception as e:
                print(f"Failed to read IMU: {e}")

    # --- GNSS Analysis ---
    print("\n--- GNSS Data Analysis (Sampled) ---")
    if gnss_dfs:
        gnss_all = pd.concat(gnss_dfs, ignore_index=True)
        print(f"GNSS Sample Shape: {gnss_all.shape}")

        gnss_num_cols = [
            "Cn0DbHz",
            "RawPseudorangeMeters",
            "PseudorangeRateMetersPerSecond",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "BiasNanos",
            "DriftNanosPerSecond",
        ]
        analyze_numerical(gnss_all, gnss_num_cols)

        print("\n--- GNSS Categorical Analysis ---")
        gnss_cat_cols = [
            "ConstellationType",
            "SignalType",
            "CodeType",
            "MultipathIndicator",
        ]
        analyze_categorical(gnss_all, gnss_cat_cols)

    else:
        print("No GNSS data loaded.")

    # --- IMU Analysis ---
    print("\n--- IMU Data Analysis (Sampled) ---")
    if imu_dfs:
        imu_all = pd.concat(imu_dfs, ignore_index=True)
        print(f"IMU Sample Shape: {imu_all.shape}")

        # IMU usually has MeasurementX, Y, Z. MessageType indicates if it's Accel, Gyro, Mag.
        # Let's analyze stats per MessageType
        for mtype in imu_all["MessageType"].unique():
            print(f"\nIMU Sensor: {mtype}")
            subset = imu_all[imu_all["MessageType"] == mtype]
            cols = ["MeasurementX", "MeasurementY", "MeasurementZ"]
            analyze_numerical(subset, cols)
    else:
        print("No IMU data loaded.")

    # ---------------------------------------------------------
    print_header("4. Feature Relationships & Importance")

    if gnss_dfs and gt_dfs:
        # We need to aggregate GNSS to match GT timestamps (1Hz)
        # GNSS 'utcTimeMillis' vs GT 'UnixTimeMillis'

        # Process one drive for relationship analysis to keep it simple and correct
        # Picking the first one from the sample list
        drive_id = sample_drives[0]

        # Re-load full data for this specific drive (without row limit if possible, or larger limit)
        # to ensure overlap in time
        row = meta_df[meta_df["drive_id"] == drive_id].iloc[0]

        gt_path = os.path.join(INPUT_DIR, row["gt_path"])
        gnss_path = os.path.join(INPUT_DIR, row["gnss_path"])

        df_gt_rel = pd.read_csv(gt_path)
        df_gnss_rel = pd.read_csv(gnss_path)  # Load full to ensure overlap

        # Aggregate GNSS by timestamp
        # Group by utcTimeMillis
        gnss_agg = df_gnss_rel.groupby("utcTimeMillis").agg(
            {
                "Cn0DbHz": ["mean", "max", "min", "std"],
                "SvElevationDegrees": ["mean"],
                "Svid": ["count"],  # Number of satellites
            }
        )

        # Flatten columns
        gnss_agg.columns = ["_".join(col).strip() for col in gnss_agg.columns.values]
        gnss_agg.reset_index(inplace=True)
        gnss_agg.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

        # Merge with GT
        # GT is target, GNSS is feature
        merged = pd.merge(df_gt_rel, gnss_agg, on="UnixTimeMillis", how="inner")

        print(f"Merged Data Shape for Drive {drive_id}: {merged.shape}")

        if len(merged) > 10:
            # Correlation Analysis
            print("\n--- Top Correlations with SpeedMps (Proxy for Dynamics) ---")
            # We use SpeedMps as a proxy target because Lat/Lon depend on absolute time/location
            # while Speed correlates with signal dynamics (Doppler, etc.)
            corr_cols = [
                c
                for c in merged.columns
                if "Cn0" in c or "Sv" in c or "Svid" in c or "Speed" in c
            ]
            if "SpeedMps" in merged.columns:
                corrs = (
                    merged[corr_cols].corr()["SpeedMps"].sort_values(ascending=False)
                )
                print(corrs.head(10))
                print("\n--- Bottom Correlations with SpeedMps ---")
                print(corrs.tail(5))

            # Feature Importance using Random Forest
            print("\n--- Feature Importance (Predicting SpeedMps) ---")
            if "SpeedMps" in merged.columns:
                feature_cols = [
                    c for c in merged.columns if "Cn0" in c or "Sv" in c or "Svid" in c
                ]
                # Drop NaNs
                dataset = merged[feature_cols + ["SpeedMps"]].dropna()

                if not dataset.empty:
                    X = dataset[feature_cols]
                    y = dataset["SpeedMps"]

                    rf = RandomForestRegressor(
                        n_estimators=50, max_depth=5, random_state=SEED
                    )
                    rf.fit(X, y)

                    importances = pd.DataFrame(
                        {"Feature": feature_cols, "Importance": rf.feature_importances_}
                    ).sort_values(by="Importance", ascending=False)

                    print(importances.head(10).to_string(index=False))
                else:
                    print("Empty dataset after dropping NaNs for RF.")
            else:
                print("SpeedMps not found in Ground Truth.")
        else:
            print("Not enough overlapping data for relationship analysis.")

    else:
        print("Skipping relationship analysis due to missing data.")


if __name__ == "__main__":
    run_eda()
