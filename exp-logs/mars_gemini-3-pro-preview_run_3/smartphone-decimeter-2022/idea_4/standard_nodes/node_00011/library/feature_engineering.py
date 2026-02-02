import pandas as pd
import numpy as np
import os
from library.config import Config
from library.data_loader import load_gnss, load_imu

# -------------------------------------------------------------------------
# WGS84 Constants for ECEF to LLA Conversion
# -------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_B = 6356752.314245
WGS84_F = (WGS84_A - WGS84_B) / WGS84_A
WGS84_E_SQ = WGS84_F * (2 - WGS84_F)
WGS84_E_SQ_PRIME = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2


def ecef_to_lla(x, y, z):
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (WGS84).
    Vectorized implementation.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + WGS84_E_SQ_PRIME * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E_SQ * WGS84_A * np.cos(theta) ** 3,
    )

    return np.degrees(lat), np.degrees(lon)


def get_wls_baseline(gnss_path):
    """
    Extracts the WLS baseline position from the raw GNSS file.
    The device_gnss.csv contains WlsPosition[X/Y/Z]EcefMeters columns.
    We convert these to Latitude/Longitude.
    """
    full_path = os.path.join(Config.INPUT_DIR, gnss_path)

    # Columns required for WLS baseline
    wls_cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    try:
        # Read specific columns
        df = pd.read_csv(full_path, usecols=wls_cols)
    except ValueError:
        # Fallback if columns missing
        return pd.DataFrame(columns=["UnixTimeMillis", "wls_lat", "wls_lon"])

    # Drop rows where WLS solution is missing
    df = df.dropna(
        subset=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    )

    if df.empty:
        return pd.DataFrame(columns=["UnixTimeMillis", "wls_lat", "wls_lon"])

    # The file contains one row per satellite per epoch.
    # The WLS position is the user position, so it is identical for all rows in the same epoch.
    # We group by timestamp and take the first available position.
    df_agg = df.groupby("utcTimeMillis").first().reset_index()

    # Convert ECEF to LLA
    lat, lon = ecef_to_lla(
        df_agg["WlsPositionXEcefMeters"].values,
        df_agg["WlsPositionYEcefMeters"].values,
        df_agg["WlsPositionZEcefMeters"].values,
    )

    df_agg["wls_lat"] = lat
    df_agg["wls_lon"] = lon

    # Rename for consistency
    return df_agg[["utcTimeMillis", "wls_lat", "wls_lon"]].rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )


def process_trip(trip_id, gnss_path, imu_path, gt_df=None):
    """
    Processes a single trip to generate windowed features and targets.

    Args:
        trip_id (str): ID of the trip.
        gnss_path (str): Relative path to GNSS file.
        imu_path (str): Relative path to IMU file.
        gt_df (pd.DataFrame, optional): Ground truth dataframe for this trip.

    Returns:
        pd.DataFrame: Processed dataframe with features and targets (if gt_df provided).
    """
    # 1. Load Raw Sensor Data
    gnss_df = load_gnss(gnss_path)
    imu_df = load_imu(imu_path)
    wls_df = get_wls_baseline(gnss_path)

    if wls_df.empty:
        return None

    # 2. Aggregate GNSS Features
    if not gnss_df.empty:
        # Group by timestamp to get 1Hz statistics
        gnss_agg = gnss_df.groupby("utcTimeMillis").agg(
            {
                "Cn0DbHz": ["mean", "std", "min", "max"],
                "SvElevationDegrees": ["mean"],
                "Svid": ["count"],
            }
        )
        # Flatten MultiIndex columns
        gnss_agg.columns = [f"gnss_{c[0]}_{c[1]}" for c in gnss_agg.columns]
        gnss_agg = gnss_agg.reset_index().rename(
            columns={"utcTimeMillis": "UnixTimeMillis"}
        )
    else:
        # Create empty with correct columns if needed, though usually GNSS is present if WLS is
        gnss_agg = pd.DataFrame(columns=["UnixTimeMillis"])

    # 3. Aggregate IMU Features
    if not imu_df.empty:
        # Filter for Accelerometer
        acc_df = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()
        if not acc_df.empty:
            # Calculate total acceleration magnitude
            acc_df["acc_mag"] = np.sqrt(
                acc_df["MeasurementX"] ** 2
                + acc_df["MeasurementY"] ** 2
                + acc_df["MeasurementZ"] ** 2
            )
            # Group by timestamp
            imu_agg = acc_df.groupby("utcTimeMillis").agg({"acc_mag": ["mean", "std"]})
            imu_agg.columns = [f"imu_{c[0]}_{c[1]}" for c in imu_agg.columns]
            imu_agg = imu_agg.reset_index().rename(
                columns={"utcTimeMillis": "UnixTimeMillis"}
            )
        else:
            imu_agg = pd.DataFrame(columns=["UnixTimeMillis"])
    else:
        imu_agg = pd.DataFrame(columns=["UnixTimeMillis"])

    # 4. Merge Data
    # Start with WLS baseline timestamps
    df = wls_df.copy()

    # Merge GNSS stats
    if not gnss_agg.empty:
        df = pd.merge(df, gnss_agg, on="UnixTimeMillis", how="left")

    # Merge IMU stats
    if not imu_agg.empty:
        df = pd.merge(df, imu_agg, on="UnixTimeMillis", how="left")

    # Fill missing values (e.g., if IMU data gaps exist)
    # Forward fill then backward fill is reasonable for sensor streams
    feature_cols = [
        c for c in df.columns if c.startswith("gnss_") or c.startswith("imu_")
    ]
    if feature_cols:
        df[feature_cols] = (
            df[feature_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)
        )

    # 5. Windowing (Temporal Context)
    # Sort by time to ensure correct shifting
    df = df.sort_values("UnixTimeMillis").reset_index(drop=True)

    window_size = Config.WINDOW_SIZE
    shifts = range(-window_size, window_size + 1)

    windowed_dfs = []

    for shift in shifts:
        # Shift features
        shifted = df[feature_cols].shift(-shift)
        # Rename columns
        suffix = f"_t{shift:+d}" if shift != 0 else "_t0"
        shifted.columns = [c + suffix for c in shifted.columns]
        windowed_dfs.append(shifted)

    # Concatenate windowed features to the main dataframe
    df_features = pd.concat(windowed_dfs, axis=1)
    df_final = pd.concat(
        [df[["UnixTimeMillis", "wls_lat", "wls_lon"]], df_features], axis=1
    )

    # Handle edges (NaNs from shifting)
    # We fill edges with the nearest valid observation
    df_final = df_final.fillna(method="ffill").fillna(method="bfill")

    # 6. Compute Targets (Training Mode)
    if gt_df is not None:
        # Merge Ground Truth
        df_final = pd.merge(
            df_final,
            gt_df[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
            on="UnixTimeMillis",
            how="inner",
        )

        # Calculate Residuals: Target = GT - WLS
        df_final["target_lat"] = df_final["LatitudeDegrees"] - df_final["wls_lat"]
        df_final["target_lon"] = df_final["LongitudeDegrees"] - df_final["wls_lon"]

    # Add Trip ID
    df_final["tripId"] = trip_id

    return df_final


def generate_dataset(metadata_df, mode="train", load_cached_data=True):
    """
    Generates the full dataset for a given split (train/val/test).
    Handles caching to Parquet files.

    Args:
        metadata_df (pd.DataFrame): Metadata containing trip info.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y) for train/val, or (X, None) for test.
    """
    # Determine cache paths
    if mode == "train":
        cache_feat = Config.TRAIN_FEATURES_PATH
        cache_targ = Config.TRAIN_TARGETS_PATH
    elif mode == "val":
        cache_feat = Config.VAL_FEATURES_PATH
        cache_targ = Config.VAL_TARGETS_PATH
    elif mode == "test":
        cache_feat = Config.TEST_FEATURES_PATH
        cache_targ = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_feat):
        print(f"Loading cached {mode} features from {cache_feat}...")
        X = pd.read_parquet(cache_feat)

        if mode != "test":
            if os.path.exists(cache_targ):
                print(f"Loading cached {mode} targets from {cache_targ}...")
                y = pd.read_parquet(cache_targ)
                return X, y
            else:
                print("Cached targets missing, regenerating...")
        else:
            return X, None

    print(f"Processing {mode} data from scratch...")

    # Debugging: Sample subset if configured
    if Config.DEBUG_SAMPLE_SIZE:
        print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} trips.")
        trips = metadata_df["tripId"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        metadata_df = metadata_df[metadata_df["tripId"].isin(trips)]

    unique_trips = metadata_df["tripId"].unique()
    results = []

    for trip_id in unique_trips:
        trip_meta = metadata_df[metadata_df["tripId"] == trip_id]
        if trip_meta.empty:
            continue

        # Get file paths from the first row of the trip metadata
        row = trip_meta.iloc[0]

        # Prepare GT dataframe for training/val
        gt_df = None
        if mode != "test":
            gt_df = trip_meta[["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

        # Process the trip
        df_trip = process_trip(
            trip_id=trip_id,
            gnss_path=row["gnss_path"],
            imu_path=row["imu_path"],
            gt_df=gt_df,
        )

        if df_trip is not None and not df_trip.empty:
            results.append(df_trip)

    if not results:
        raise RuntimeError(f"No data generated for mode {mode}. Check input paths.")

    full_df = pd.concat(results, ignore_index=True)

    # Separate Features and Targets
    # Features include windowed sensor stats
    feature_cols = [c for c in full_df.columns if "gnss_" in c or "imu_" in c]
    # Meta columns needed for post-processing/evaluation
    meta_cols = ["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]

    X = full_df[meta_cols + feature_cols]

    if mode != "test":
        y = full_df[["target_lat", "target_lon"]]

        # Save to cache
        print(f"Saving {mode} data to cache...")
        X.to_parquet(cache_feat)
        y.to_parquet(cache_targ)

        return X, y
    else:
        # For test, we only save features
        print(f"Saving {mode} data to cache...")
        X.to_parquet(cache_feat)
        return X, None
