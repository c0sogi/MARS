import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import load_metadata
from library.coord_utils import ecef_to_wgs84, geodetic_to_enu


def extract_gnss_features(gnss_df):
    """
    Extract point-wise features from GNSS data.
    Aggregates raw signals by timestamp (1Hz).
    Includes signal diversity features (L1 vs L5 split).
    """
    # Define L5 signal types (others are considered L1-like)
    l5_signals = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]

    # Create binary flag for L5
    gnss_df["is_l5"] = gnss_df["SignalType"].isin(l5_signals)

    # Base aggregation dictionary
    agg_dict = {
        "Svid": "count",
        "Cn0DbHz": ["mean", "std", "max"],
        "SvElevationDegrees": "mean",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Group by timestamp
    grouped = gnss_df.groupby("utcTimeMillis").agg(agg_dict)

    # Flatten columns
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

    # Rename base features
    rename_map = {
        "Svid_count": "sv_count",
        "Cn0DbHz_mean": "cn0_mean",
        "Cn0DbHz_std": "cn0_std",
        "Cn0DbHz_max": "cn0_max",
        "SvElevationDegrees_mean": "elev_mean",
        "WlsPositionXEcefMeters_first": "wls_x",
        "WlsPositionYEcefMeters_first": "wls_y",
        "WlsPositionZEcefMeters_first": "wls_z",
    }
    grouped.rename(columns=rename_map, inplace=True)

    # --- L1 vs L5 Specific Features ---
    # Pivot or separate groupby to get L1/L5 specific counts and signal strengths

    # L5 Features
    l5_df = gnss_df[gnss_df["is_l5"]]
    if not l5_df.empty:
        l5_agg = l5_df.groupby("utcTimeMillis").agg(
            {"Svid": "count", "Cn0DbHz": "mean"}
        )
        l5_agg.columns = ["l5_count", "l5_cn0_mean"]
    else:
        l5_agg = pd.DataFrame(columns=["l5_count", "l5_cn0_mean"])
        l5_agg.index.name = "utcTimeMillis"

    # L1 Features (Inverse of L5)
    l1_df = gnss_df[~gnss_df["is_l5"]]
    if not l1_df.empty:
        l1_agg = l1_df.groupby("utcTimeMillis").agg(
            {"Svid": "count", "Cn0DbHz": "mean"}
        )
        l1_agg.columns = ["l1_count", "l1_cn0_mean"]
    else:
        l1_agg = pd.DataFrame(columns=["l1_count", "l1_cn0_mean"])
        l1_agg.index.name = "utcTimeMillis"

    # Merge specific features back to main grouped df
    grouped = grouped.join(l5_agg, how="left")
    grouped = grouped.join(l1_agg, how="left")

    # Fill NaNs for counts with 0 (if no signals of that type found for a timestamp)
    grouped["l5_count"] = grouped["l5_count"].fillna(0)
    grouped["l1_count"] = grouped["l1_count"].fillna(0)

    # Reset index to make UnixTimeMillis a column
    grouped.reset_index(inplace=True)
    grouped.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    return grouped


def extract_imu_features(imu_df):
    """
    Extract point-wise features from IMU data (Accelerometer).
    Aligns high-frequency IMU data to 1Hz GNSS timestamps.
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
    # utcTimeMillis is int, round to nearest 1000
    accel["UnixTimeMillis"] = (np.round(accel["utcTimeMillis"] / 1000) * 1000).astype(
        np.int64
    )

    # Group and aggregate
    grouped = (
        accel.groupby("UnixTimeMillis")["magnitude"].agg(["mean", "std"]).reset_index()
    )
    grouped.rename(columns={"mean": "accel_mean", "std": "accel_std"}, inplace=True)

    return grouped


def process_drive(drive_id, phone_name, df_meta_drive, input_dir, split):
    """
    Process a single drive: load raw files, extract physics features, merge, and compute targets.
    """
    # Get file paths from the first row of metadata for this drive
    first_row = df_meta_drive.iloc[0]

    gnss_path = os.path.join(input_dir, first_row["gnss_path"])
    imu_path = os.path.join(input_dir, first_row["imu_path"])

    # Load and Process GNSS
    if os.path.exists(gnss_path):
        gnss_df = pd.read_csv(gnss_path)
        features = extract_gnss_features(gnss_df)
    else:
        return pd.DataFrame()

    # Load and Process IMU
    if os.path.exists(imu_path):
        imu_df = pd.read_csv(imu_path)
        imu_features = extract_imu_features(imu_df)
        # Merge IMU features
        features = pd.merge(features, imu_features, on="UnixTimeMillis", how="left")
    else:
        # Add missing columns if IMU missing
        features["accel_mean"] = np.nan
        features["accel_std"] = np.nan

    # Merge with Metadata
    # Metadata contains the ground truth (for train/val) and the required timestamps
    merged = pd.merge(df_meta_drive, features, on="UnixTimeMillis", how="inner")

    # Compute Baseline WLS Geodetic Coordinates
    wls_x = merged["wls_x"].values
    wls_y = merged["wls_y"].values
    wls_z = merged["wls_z"].values

    wls_lat, wls_lon, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

    merged["wls_lat"] = wls_lat
    merged["wls_lon"] = wls_lon
    merged["wls_alt"] = wls_alt

    # Compute Targets (ENU Residuals) only for Train/Val
    # For Test, we do not compute targets even if columns exist (they are placeholders)
    if split in ["train", "val"]:
        gt_lat = merged["LatitudeDegrees"].values
        gt_lon = merged["LongitudeDegrees"].values

        # Calculate ENU residuals: Vector from WLS to GT
        # We use WLS altitude for both to isolate horizontal error
        e, n, u = geodetic_to_enu(gt_lat, gt_lon, wls_alt, wls_lat, wls_lon, wls_alt)

        merged["target_E"] = e
        merged["target_N"] = n

    return merged


def get_data(split, load_cached_data=True):
    """
    Main function to get the dataset for a split.
    Handles caching, feature extraction, and aggregation.
    """
    # Determine cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_features.parquet")

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
        drives = meta_df["drive_id"].unique()[:2]
        meta_df = meta_df[meta_df["drive_id"].isin(drives)].copy()

    # Group by drive
    grouped = meta_df.groupby(["drive_id", "phone_name"])

    processed_dfs = []

    # Iterate without tqdm to satisfy requirements
    total_groups = len(grouped)
    print(f"Processing {total_groups} drives for split '{split}'...")

    for i, ((drive_id, phone_name), group) in enumerate(grouped):
        df_drive = process_drive(drive_id, phone_name, group, Config.INPUT_DIR, split)
        if not df_drive.empty:
            processed_dfs.append(df_drive)

    if not processed_dfs:
        raise ValueError(f"No data processed for split {split}")

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    full_df.to_parquet(cache_path, index=False)

    return full_df
