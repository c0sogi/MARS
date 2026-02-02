import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger
from library.coords import ecef_to_geodetic, geodetic_to_ecef, ecef_to_enu
import library.data_loader as data_loader

logger = get_logger(__name__)


def process_gnss(gnss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates GNSS data by epoch to create signal and geometry features.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS dataframe.

    Returns:
        pd.DataFrame: Aggregated features per epoch.
    """
    # Create masks for signal frequencies
    # L1: GPS L1, GAL E1, GLO G1, etc.
    gnss_df["is_L1"] = (
        gnss_df["SignalType"]
        .astype(str)
        .apply(lambda x: "1" in x or "G1" in x or "E1" in x)
    )
    # L5: GPS L5, GAL E5a, etc.
    gnss_df["is_L5"] = (
        gnss_df["SignalType"].astype(str).apply(lambda x: "5" in x or "2A" in x)
    )

    # Filter out invalid signal strengths
    gnss_df = gnss_df[gnss_df["Cn0DbHz"].notna() & (gnss_df["Cn0DbHz"] > 0)].copy()

    # Define aggregation dictionary
    aggs = {
        "Cn0DbHz": ["mean", "max", "std"],
        "SvElevationDegrees": ["mean"],
        "Svid": ["count"],
        # WLS positions are repeated per signal in an epoch, take first
        "WlsPositionXEcefMeters": ["first"],
        "WlsPositionYEcefMeters": ["first"],
        "WlsPositionZEcefMeters": ["first"],
    }

    # If Ground Truth columns exist (train/val splits), preserve them
    if "LatitudeDegrees" in gnss_df.columns:
        aggs["LatitudeDegrees"] = ["first"]
        aggs["LongitudeDegrees"] = ["first"]
    if "AltitudeMeters" in gnss_df.columns:
        aggs["AltitudeMeters"] = ["first"]

    # Group by unique epoch identifier (drive + phone + time)
    grouped = gnss_df.groupby(["drive_id", "phone_name", "utcTimeMillis"])

    # Perform aggregation
    df_agg = grouped.agg(aggs)

    # Flatten MultiIndex columns
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]

    # Rename columns to match Config features
    rename_map = {
        "Cn0DbHz_mean": "Cn0DbHz_mean",
        "Cn0DbHz_max": "Cn0DbHz_max",
        "Cn0DbHz_std": "Cn0DbHz_std",
        "SvElevationDegrees_mean": "SvElevationDegrees_mean",
        "Svid_count": "sv_count",
        "WlsPositionXEcefMeters_first": "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters_first": "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters_first": "WlsPositionZEcefMeters",
        "LatitudeDegrees_first": "LatitudeDegrees",
        "LongitudeDegrees_first": "LongitudeDegrees",
        "AltitudeMeters_first": "AltitudeMeters",
    }
    df_agg = df_agg.rename(columns=rename_map)

    # Calculate specific L1/L5 mean signal strength manually
    # This avoids complex conditional aggregations in the main groupby
    l1_mean = (
        gnss_df[gnss_df["is_L1"]]
        .groupby(["drive_id", "phone_name", "utcTimeMillis"])["Cn0DbHz"]
        .mean()
    )
    l5_mean = (
        gnss_df[gnss_df["is_L5"]]
        .groupby(["drive_id", "phone_name", "utcTimeMillis"])["Cn0DbHz"]
        .mean()
    )

    # Join these back to the aggregated dataframe
    # We reset index temporarily to join, or assign directly if indices align (they should)
    df_agg["Cn0DbHz_L1_mean"] = l1_mean
    df_agg["Cn0DbHz_L5_mean"] = l5_mean

    # Fill NaNs for epochs that might lack L1 or L5 signals entirely
    df_agg["Cn0DbHz_L1_mean"] = df_agg["Cn0DbHz_L1_mean"].fillna(0)
    df_agg["Cn0DbHz_L5_mean"] = df_agg["Cn0DbHz_L5_mean"].fillna(0)

    # Reset index to make grouping keys available as columns
    df_agg = df_agg.reset_index()

    return df_agg


def process_imu(imu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates IMU accelerometer data by epoch.

    Args:
        imu_df (pd.DataFrame): Raw IMU dataframe.

    Returns:
        pd.DataFrame: Aggregated IMU features per epoch.
    """
    # Filter for Uncalibrated Accelerometer data
    accel = imu_df[imu_df["MessageType"] == "UncalAccel"].copy()

    if accel.empty:
        # Return empty DataFrame with expected columns if no accel data exists
        return pd.DataFrame(
            columns=[
                "drive_id",
                "phone_name",
                "utcTimeMillis",
                "accel_mag_mean",
                "accel_mag_std",
            ]
        )

    # Calculate magnitude of the acceleration vector
    accel["mag"] = np.sqrt(
        accel["MeasurementX"] ** 2
        + accel["MeasurementY"] ** 2
        + accel["MeasurementZ"] ** 2
    )

    # Group by epoch
    grouped = accel.groupby(["drive_id", "phone_name", "utcTimeMillis"])

    # Aggregate
    df_agg = grouped.agg({"mag": ["mean", "std"]})

    # Flatten and rename
    df_agg.columns = ["accel_mag_mean", "accel_mag_std"]
    df_agg = df_agg.reset_index()

    return df_agg


def create_features(gnss_df: pd.DataFrame, imu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates feature creation by processing GNSS and IMU data and merging them.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data.
        imu_df (pd.DataFrame): Raw IMU data.

    Returns:
        pd.DataFrame: Feature dataframe ready for model training/inference.
    """
    logger.info("Processing GNSS data...")
    gnss_feats = process_gnss(gnss_df)

    logger.info("Processing IMU data...")
    imu_feats = process_imu(imu_df)

    logger.info("Merging features...")
    # Merge GNSS and IMU features
    # We use a left join on GNSS epochs because GNSS determines the prediction timestamps
    merged = pd.merge(
        gnss_feats,
        imu_feats,
        on=["drive_id", "phone_name", "utcTimeMillis"],
        how="left",
    )

    # Fill missing IMU data (e.g., if IMU frequency is lower or gaps exist)
    # Mean magnitude ~ 9.8 m/s^2 (gravity), std ~ 0 (static)
    merged["accel_mag_mean"] = merged["accel_mag_mean"].fillna(9.8)
    merged["accel_mag_std"] = merged["accel_mag_std"].fillna(0)

    # Create tripId for submission/evaluation grouping
    merged["tripId"] = merged["drive_id"] + "-" + merged["phone_name"]

    # Rename timestamp column to match submission format
    merged = merged.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

    return merged


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes regression targets (East/North errors) in the ENU frame.

    Args:
        df (pd.DataFrame): Dataframe containing WLS positions and Ground Truth.

    Returns:
        pd.DataFrame: Dataframe with added target columns.
    """
    # Ensure Ground Truth columns exist
    if "LatitudeDegrees" not in df.columns or "LongitudeDegrees" not in df.columns:
        raise ValueError("Ground Truth columns missing for target computation")

    # 1. Convert WLS Baseline ECEF to Geodetic
    # These geodetic coordinates will serve as the reference point for the local ENU frame
    # Drop rows where WLS is missing (cannot compute residual)
    df = df.dropna(
        subset=[
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    ).copy()

    wls_x = df["WlsPositionXEcefMeters"].values
    wls_y = df["WlsPositionYEcefMeters"].values
    wls_z = df["WlsPositionZEcefMeters"].values

    ref_lat, ref_lon, ref_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # 2. Convert Ground Truth Geodetic to ECEF
    gt_lat = df["LatitudeDegrees"].values
    gt_lon = df["LongitudeDegrees"].values
    # Handle missing GT Altitude: Fill with Ref Altitude (WLS) to minimize vertical noise impact
    gt_alt = df["AltitudeMeters"].fillna(pd.Series(ref_alt)).values

    gt_x, gt_y, gt_z = geodetic_to_ecef(gt_lat, gt_lon, gt_alt)

    # 3. Compute ENU Residuals
    # Vector = GT - WLS (The correction needed to get from WLS to GT)
    # Note: ecef_to_enu calculates vector from Ref to Target.
    # Here Ref=WLS, Target=GT. So result is (GT - WLS) in ENU.
    d_east, d_north, d_up = ecef_to_enu(gt_x, gt_y, gt_z, ref_lat, ref_lon, ref_alt)

    df[Config.TARGET_EAST] = d_east
    df[Config.TARGET_NORTH] = d_north

    # Store reference coordinates needed for reconstruction
    df["RefLat"] = ref_lat
    df["RefLon"] = ref_lon
    df["RefAlt"] = ref_alt

    return df


def prepare_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main entry point for data preparation. Loads raw data, computes features/targets, and caches results.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe ready for the model.
    """
    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_features.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {split} features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    logger.info(f"Computing {split} features from scratch...")

    # Load raw data (data_loader handles raw caching)
    gnss_raw, imu_raw = data_loader.load_split_data(
        split, load_cached_data=load_cached_data
    )

    # Feature Engineering
    df_features = create_features(gnss_raw, imu_raw)

    # Target Computation (Train/Val only)
    if split in ["train", "val"]:
        logger.info("Computing targets...")
        df_features = compute_targets(df_features)
    else:
        # For Test, we still need Reference Coordinates (WLS) to reconstruct predictions later
        # We perform the WLS ECEF -> Geodetic conversion here
        logger.info("Computing reference coordinates for test set...")
        wls_x = df_features["WlsPositionXEcefMeters"].values
        wls_y = df_features["WlsPositionYEcefMeters"].values
        wls_z = df_features["WlsPositionZEcefMeters"].values
        ref_lat, ref_lon, ref_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)
        df_features["RefLat"] = ref_lat
        df_features["RefLon"] = ref_lon
        df_features["RefAlt"] = ref_alt

    # 3. Save to cache
    logger.info(f"Saving {split} features to cache: {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
