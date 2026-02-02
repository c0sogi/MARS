import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.coord_utils import ecef_to_wgs84, geodetic_to_enu


def load_metadata(split):
    """
    Load metadata for a specific split (train, val, test).
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df


def aggregate_gnss(gnss_df):
    """
    Aggregate raw GNSS signals to 1Hz epoch level features.
    """
    # Filter out invalid signals if necessary, but usually we use all available in device_gnss
    # We assume utcTimeMillis is the join key

    # Define aggregation dictionary
    agg_dict = {
        "Svid": "count",
        "Cn0DbHz": ["mean", "std", "max"],
        "SvElevationDegrees": "mean",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Group by timestamp
    # Note: device_gnss.csv usually has utcTimeMillis.
    grouped = gnss_df.groupby("utcTimeMillis").agg(agg_dict)

    # Flatten column names
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)

    # Rename columns for clarity
    grouped.rename(
        columns={
            "utcTimeMillis": "UnixTimeMillis",
            "Svid_count": "sv_count",
            "Cn0DbHz_mean": "cn0_mean",
            "Cn0DbHz_std": "cn0_std",
            "Cn0DbHz_max": "cn0_max",
            "SvElevationDegrees_mean": "elev_mean",
            "WlsPositionXEcefMeters_first": "wls_x",
            "WlsPositionYEcefMeters_first": "wls_y",
            "WlsPositionZEcefMeters_first": "wls_z",
        },
        inplace=True,
    )

    return grouped


def aggregate_imu(imu_df):
    """
    Aggregate IMU data to 1Hz.
    IMU data is high frequency (~100Hz). We align it to the nearest second.
    """
    # Filter for Accelerometer
    accel = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()

    if accel.empty:
        return pd.DataFrame(columns=["UnixTimeMillis", "accel_mean", "accel_std"])

    # Calculate magnitude
    accel["magnitude"] = np.sqrt(
        accel["MeasurementX"] ** 2
        + accel["MeasurementY"] ** 2
        + accel["MeasurementZ"] ** 2
    )

    # Round timestamp to nearest second to match GNSS 1Hz
    # utcTimeMillis is int
    accel["UnixTimeMillis"] = (np.round(accel["utcTimeMillis"] / 1000) * 1000).astype(
        np.int64
    )

    # Group
    grouped = (
        accel.groupby("UnixTimeMillis")["magnitude"].agg(["mean", "std"]).reset_index()
    )
    grouped.rename(columns={"mean": "accel_mean", "std": "accel_std"}, inplace=True)

    return grouped


def process_drive(drive_id, phone_name, df_meta_drive, input_dir):
    """
    Process a single drive: load raw files, aggregate, merge, and compute targets.
    """
    # Get file paths from the first row of metadata for this drive
    # Paths in metadata are relative to input_dir
    first_row = df_meta_drive.iloc[0]

    gnss_path = os.path.join(input_dir, first_row["gnss_path"])
    imu_path = os.path.join(input_dir, first_row["imu_path"])

    # Load GNSS
    if os.path.exists(gnss_path):
        gnss_df = pd.read_csv(gnss_path)
        gnss_agg = aggregate_gnss(gnss_df)
    else:
        # Should not happen based on metadata checks, but handle gracefully
        return pd.DataFrame()

    # Load IMU
    if os.path.exists(imu_path):
        imu_df = pd.read_csv(imu_path)
        imu_agg = aggregate_imu(imu_df)
    else:
        imu_agg = pd.DataFrame(columns=["UnixTimeMillis", "accel_mean", "accel_std"])

    # Merge Features (GNSS + IMU)
    # We use GNSS timestamps as the base
    features = pd.merge(gnss_agg, imu_agg, on="UnixTimeMillis", how="left")

    # Fill missing IMU data (if any) with defaults or NaN (LightGBM handles NaN)

    # Merge with Metadata (which contains GT for train/val, or required timestamps for test)
    # Metadata has 'UnixTimeMillis'
    merged = pd.merge(df_meta_drive, features, on="UnixTimeMillis", how="inner")

    # Compute Baseline WLS Geodetic Coordinates
    # We need these to compute residuals and for final reconstruction
    # Vectorized conversion
    wls_x = merged["wls_x"].values
    wls_y = merged["wls_y"].values
    wls_z = merged["wls_z"].values

    wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

    merged["wls_lat"] = wls_lat
    merged["wls_lon"] = wls_lon
    merged["wls_alt"] = wls_alt

    # Compute Targets (if GT is available)
    if "LatitudeDegrees" in merged.columns and "LongitudeDegrees" in merged.columns:
        # For Test set, these columns might exist in sample submission but shouldn't be used as targets.
        # We assume if it's called from 'train' or 'val' split logic, we compute targets.
        # We check if the values are not all the same placeholder (sample submission often has placeholders)
        # But relying on split name passed to get_processed_dataset is safer.
        # Here we just compute if columns exist, caller decides usage.

        gt_lat = merged["LatitudeDegrees"].values
        gt_lon = merged["LongitudeDegrees"].values

        # We assume GT altitude is same as WLS altitude for horizontal error calculation
        # to avoid noise from vertical component which is not scored.
        # Calculate ENU residuals: Vector from WLS to GT
        e, n, u = geodetic_to_enu(gt_lat, gt_lon, wls_alt, wls_lat, wls_lon, wls_alt)

        merged["target_E"] = e
        merged["target_N"] = n

    return merged


def get_processed_dataset(split, load_cached_data=True):
    """
    Main function to get the dataset for a split.
    Handles caching and aggregation.
    """
    # Determine cache path
    if split == "train":
        cache_path = Config.TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = Config.VAL_FEATURES_PATH
    elif split == "test":
        cache_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {split} data from raw files...")
    meta_df = load_metadata(split)

    if Config.DEBUG:
        print("DEBUG Mode: Sampling metadata...")
        drives = meta_df["drive_id"].unique()[:2]  # Take first 2 drives
        meta_df = meta_df[meta_df["drive_id"].isin(drives)].copy()

    # Group by drive to process file-by-file
    grouped = meta_df.groupby(["drive_id", "phone_name"])

    processed_dfs = []

    # Iterate with progress bar
    for (drive_id, phone_name), group in tqdm(
        grouped, desc=f"Processing {split} drives"
    ):
        df_drive = process_drive(drive_id, phone_name, group, Config.INPUT_DIR)
        if not df_drive.empty:
            processed_dfs.append(df_drive)

    if not processed_dfs:
        raise ValueError(f"No data processed for split {split}")

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    full_df.to_parquet(cache_path, index=False)

    return full_df
