import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    GNSS_COLS,
    IMU_COLS,
    SEED,
)
from library.utils import wgs84_to_enu, ecef_to_wgs84

# Set random seed
np.random.seed(SEED)


def load_metadata(mode="train"):
    """
    Load the metadata CSV file for the specified mode.

    Args:
        mode (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if mode == "train":
        path = TRAIN_METADATA_PATH
    elif mode == "val":
        path = VAL_METADATA_PATH
    elif mode == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def _process_trip_gnss(gnss_path):
    """
    Load and aggregate GNSS data for a single trip.
    """
    full_path = os.path.join(INPUT_DIR, gnss_path)
    if not os.path.exists(full_path):
        return None

    # Load specific columns
    try:
        df = pd.read_csv(full_path, usecols=lambda c: c in GNSS_COLS)
    except ValueError:
        # Fallback if some columns are missing (rare but possible in test set structure variations)
        df = pd.read_csv(full_path)
        missing_cols = [c for c in GNSS_COLS if c not in df.columns]
        if missing_cols:
            # If critical columns missing, return None or empty
            return None

    # Rename time column for consistency
    df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # 1. Extract WLS Position (Baseline)
    # WLS position is repeated for each satellite in the same epoch. Take the first one.
    wls_cols = [
        "UnixTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df_wls = df[wls_cols].drop_duplicates(subset=["UnixTimeMillis"]).copy()

    # Convert WLS ECEF to Geodetic
    # Vectorized conversion
    lats, lons, alts = ecef_to_wgs84(
        df_wls["WlsPositionXEcefMeters"].values,
        df_wls["WlsPositionYEcefMeters"].values,
        df_wls["WlsPositionZEcefMeters"].values,
    )
    df_wls["WlsLat"] = lats
    df_wls["WlsLon"] = lons
    df_wls["WlsAlt"] = alts

    # Drop ECEF columns to save space
    df_wls.drop(
        columns=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ],
        inplace=True,
    )

    # 2. Aggregate Signal Features
    # Group by timestamp
    grouped = df.groupby("UnixTimeMillis")

    agg_funcs = {
        "Cn0DbHz": ["mean", "max", "std"],
        "SvElevationDegrees": ["mean"],
        "Svid": ["count"],
    }

    df_feats = grouped.agg(agg_funcs)

    # Flatten MultiIndex columns
    df_feats.columns = [f"{col[0]}_{col[1]}" for col in df_feats.columns]
    df_feats.reset_index(inplace=True)

    # Rename for clarity
    df_feats.rename(columns={"Svid_count": "Svid_count"}, inplace=True)

    # Merge WLS and Features
    df_merged = pd.merge(df_wls, df_feats, on="UnixTimeMillis", how="inner")

    return df_merged


def _process_trip_imu(imu_path):
    """
    Load and aggregate IMU data for a single trip.
    """
    full_path = os.path.join(INPUT_DIR, imu_path)
    if not os.path.exists(full_path):
        return None

    try:
        df = pd.read_csv(full_path, usecols=lambda c: c in IMU_COLS)
    except ValueError:
        return None

    # Filter for Accelerometer only (UncalAccel is standard for this dataset)
    df = df[df["MessageType"] == "UncalAccel"].copy()

    if df.empty:
        return None

    # Calculate Magnitude
    df["Accel_Mag"] = np.sqrt(
        df["MeasurementX"] ** 2 + df["MeasurementY"] ** 2 + df["MeasurementZ"] ** 2
    )

    # IMU is high frequency. We align to GNSS seconds.
    # Round timestamp to nearest second (1000 ms)
    # Note: utcTimeMillis in IMU is derived and usually close to GNSS time.
    df["UnixTimeMillis"] = df["utcTimeMillis"].apply(lambda x: round(x / 1000) * 1000)

    # Aggregate
    grouped = df.groupby("UnixTimeMillis")
    df_agg = grouped.agg({"Accel_Mag": ["mean", "std"]})

    df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
    df_agg.reset_index(inplace=True)

    return df_agg


def _process_trip(row, include_gt=True):
    """
    Process a single trip: Load GNSS, IMU, and optionally Ground Truth.
    Merge and compute targets.
    """
    trip_id = row["tripId"]

    # 1. Process GNSS (Features + Baseline)
    df_gnss = _process_trip_gnss(row["gnss_path"])
    if df_gnss is None or df_gnss.empty:
        return None

    # 2. Process IMU (Features)
    df_imu = _process_trip_imu(row["imu_path"])

    # 3. Merge GNSS and IMU
    # Left join on GNSS timestamps because we need to predict for GNSS epochs
    if df_imu is not None and not df_imu.empty:
        df_features = pd.merge(df_gnss, df_imu, on="UnixTimeMillis", how="left")
        # Fill missing IMU data (if any gaps) with 0 or mean
        df_features.fillna({"Accel_Mag_mean": 0, "Accel_Mag_std": 0}, inplace=True)
    else:
        df_features = df_gnss
        df_features["Accel_Mag_mean"] = 0.0
        df_features["Accel_Mag_std"] = 0.0

    # 4. Handle Ground Truth (Targets)
    if include_gt:
        gt_path = os.path.join(INPUT_DIR, row["gt_path"])
        if not os.path.exists(gt_path):
            return None

        df_gt = pd.read_csv(gt_path)

        # Merge Features with GT
        # Inner join: we only train on epochs where we have both features and GT
        df_merged = pd.merge(
            df_features, df_gt, on="UnixTimeMillis", suffixes=("", "_gt")
        )

        # Calculate Targets (ENU Residuals)
        # Target = GT - WLS (in ENU frame)
        # We use WLS position as the reference point for ENU conversion

        # Ensure GT has Altitude. If not, fill with WLS Altitude (approximate, but affects Up mostly)
        if "AltitudeMeters" not in df_merged.columns:
            # Fallback if GT file doesn't have Altitude (unlikely for this dataset)
            df_merged["AltitudeMeters"] = df_merged["WlsAlt"]

        e, n, u = wgs84_to_enu(
            df_merged["LatitudeDegrees"].values,
            df_merged["LongitudeDegrees"].values,
            df_merged["AltitudeMeters"].values,
            df_merged["WlsLat"].values,
            df_merged["WlsLon"].values,
            df_merged["WlsAlt"].values,
        )

        df_merged["target_east"] = e
        df_merged["target_north"] = n

        # Keep metadata for grouping
        df_merged["drive_id"] = row["drive_id"]
        df_merged["phone_name"] = row["phone_name"]
        df_merged["tripId"] = trip_id

        return df_merged
    else:
        # Test mode: No GT, no targets
        df_features["tripId"] = trip_id
        df_features["drive_id"] = row["drive_id"]
        df_features["phone_name"] = row["phone_name"]
        return df_features


def prepare_training_data(load_cached_data=True):
    """
    Generate or load training data (Train + Val splits).

    Args:
        load_cached_data (bool): If True, try to load from parquet cache.

    Returns:
        tuple: (train_df, val_df)
    """
    # Check Cache
    if (
        load_cached_data
        and os.path.exists(TRAIN_FEATURES_PATH)
        and os.path.exists(VAL_FEATURES_PATH)
    ):
        print("Loading training data from cache...")
        train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
        val_df = pd.read_parquet(VAL_FEATURES_PATH)
        return train_df, val_df

    print("Generating training data from raw files...")

    # Load Metadata
    meta_train = load_metadata("train")
    meta_val = load_metadata("val")

    # Process Train
    train_dfs = []
    # Group by trip to process
    unique_trips_train = meta_train.drop_duplicates(subset=["tripId"])
    print(f"Processing {len(unique_trips_train)} training trips...")

    for _, row in unique_trips_train.iterrows():
        df = _process_trip(row, include_gt=True)
        if df is not None:
            train_dfs.append(df)

    if not train_dfs:
        raise RuntimeError("Failed to process any training trips.")

    train_df = pd.concat(train_dfs, ignore_index=True)

    # Process Val
    val_dfs = []
    unique_trips_val = meta_val.drop_duplicates(subset=["tripId"])
    print(f"Processing {len(unique_trips_val)} validation trips...")

    for _, row in unique_trips_val.iterrows():
        df = _process_trip(row, include_gt=True)
        if df is not None:
            val_dfs.append(df)

    if not val_dfs:
        raise RuntimeError("Failed to process any validation trips.")

    val_df = pd.concat(val_dfs, ignore_index=True)

    # Save to Cache
    print("Saving training data to cache...")
    train_df.to_parquet(TRAIN_FEATURES_PATH, index=False)
    val_df.to_parquet(VAL_FEATURES_PATH, index=False)

    return train_df, val_df


def prepare_test_data(load_cached_data=True):
    """
    Generate or load test data features.

    Args:
        load_cached_data (bool): If True, try to load from parquet cache.

    Returns:
        pd.DataFrame: Test features.
    """
    if load_cached_data and os.path.exists(TEST_FEATURES_PATH):
        print("Loading test data from cache...")
        return pd.read_parquet(TEST_FEATURES_PATH)

    print("Generating test data from raw files...")

    meta_test = load_metadata("test")

    # The test metadata contains all timestamps we need to predict.
    # However, we process by trip to be efficient.
    unique_trips_test = meta_test.drop_duplicates(subset=["tripId"])
    print(f"Processing {len(unique_trips_test)} test trips...")

    test_dfs = []
    for _, row in unique_trips_test.iterrows():
        df = _process_trip(row, include_gt=False)
        if df is not None:
            # Filter to only keep rows requested in sample submission
            # The meta_test has the specific timestamps for this trip
            target_timestamps = meta_test[meta_test["tripId"] == row["tripId"]][
                "UnixTimeMillis"
            ].values
            df_filtered = df[df["UnixTimeMillis"].isin(target_timestamps)].copy()
            test_dfs.append(df_filtered)

    if not test_dfs:
        raise RuntimeError("Failed to process any test trips.")

    test_df = pd.concat(test_dfs, ignore_index=True)

    # Save to Cache
    print("Saving test data to cache...")
    test_df.to_parquet(TEST_FEATURES_PATH, index=False)

    return test_df
