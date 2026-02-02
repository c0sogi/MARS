import os
import numpy as np
import pandas as pd
import math
from library.config import Config
from library.utils import WGS84Utils, get_logger

logger = get_logger("features")


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered Earth-Fixed (ECEF) coordinates to Geodetic (Lat, Lon, Alt).
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a**2
    esq = e**2

    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th) ** 3), (p - esq * a * np.cos(th) ** 3))

    # Convert radians to degrees
    return np.degrees(lat), np.degrees(lon)


def process_drive(drive_id, phone_name, gnss_path, gt_df=None):
    """
    Process a single drive's GNSS data into a feature sequence.
    """
    if not os.path.exists(gnss_path):
        logger.warning(f"GNSS file not found: {gnss_path}")
        return None

    try:
        # Load raw GNSS data
        # We need WLS columns which might not be in Config.RAW_GNSS_COLS, so we read all or append them
        # Reading specific columns + WLS columns to be efficient
        cols_to_read = list(
            set(
                Config.RAW_GNSS_COLS
                + [
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            )
        )
        gnss_df = pd.read_csv(gnss_path, usecols=lambda c: c in cols_to_read)
    except Exception as e:
        logger.error(f"Error reading {gnss_path}: {e}")
        return None

    # Filter invalid signals
    # Cn0DbHz must be positive
    gnss_df = gnss_df[gnss_df["Cn0DbHz"] > 0].copy()

    if gnss_df.empty:
        return None

    # Temporal Quantization: Align to 1Hz
    # utcTimeMillis is roughly GPS time. We round to nearest second.
    gnss_df["Epoch"] = (
        np.round(gnss_df["utcTimeMillis"] / 1000.0).astype(np.int64) * 1000
    )

    # --- Feature Engineering ---

    # 1. Pre-compute angular components (Degrees to Radians)
    az_rad = np.radians(gnss_df["SvAzimuthDegrees"].fillna(0))
    el_rad = np.radians(gnss_df["SvElevationDegrees"].fillna(0))

    # Satellite unit vectors in local frame (approximated)
    # x=East, y=North, z=Up
    gnss_df["sv_x"] = np.cos(el_rad) * np.sin(az_rad)
    gnss_df["sv_y"] = np.cos(el_rad) * np.cos(az_rad)
    gnss_df["sv_z"] = np.sin(el_rad)

    # Linear signal power for weighting (10^(dB/10))
    gnss_df["signal_power"] = 10 ** (gnss_df["Cn0DbHz"] / 10.0)

    # Weighted components
    gnss_df["w_x"] = gnss_df["sv_x"] * gnss_df["signal_power"]
    gnss_df["w_y"] = gnss_df["sv_y"] * gnss_df["signal_power"]
    gnss_df["w_z"] = gnss_df["sv_z"] * gnss_df["signal_power"]

    # Pre-compute sin/cos for angular means
    gnss_df["sin_az"] = np.sin(az_rad)
    gnss_df["cos_az"] = np.cos(az_rad)

    # 2. Aggregations per Epoch
    aggs = {
        "Cn0DbHz": ["mean", "std", "min", "max"],
        "SvElevationDegrees": ["mean", "std", "min", "max"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "Svid": ["count"],  # This becomes SatCount
        "signal_power": ["sum"],
        "w_x": ["sum"],
        "w_y": ["sum"],
        "w_z": ["sum"],
        "sin_az": ["mean"],
        "cos_az": ["mean"],
    }

    # Group by Epoch
    grouped = gnss_df.groupby("Epoch")
    features = grouped.agg(aggs)

    # Flatten MultiIndex columns (e.g., ('Cn0DbHz', 'mean') -> 'Cn0DbHz_mean')
    features.columns = ["_".join(col).strip() for col in features.columns.values]

    # Get WLS positions (Baseline)
    # These are repeated per signal in the same epoch, so we take the first one
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls_pos = grouped[wls_cols].first()

    # Join WLS to features
    features = features.join(wls_pos)

    # Rename columns to match Config.FEATURE_COLS
    rename_map = {
        "Svid_count": "SatCount",
        "sin_az_mean": "SvAzimuth_sin_mean",
        "cos_az_mean": "SvAzimuth_cos_mean",
    }
    features.rename(columns=rename_map, inplace=True)

    # 3. Compute Signal Weighted Geometry Features
    # Weighted Vector Sum / Total Power
    # Avoid division by zero
    p_sum = features["signal_power_sum"].replace(0, 1e-9)
    wx = features["w_x_sum"] / p_sum
    wy = features["w_y_sum"] / p_sum
    wz = features["w_z_sum"] / p_sum

    # Weighted Azimuth: atan2(x, y)
    features["SignalWeighted_Azimuth_rad"] = np.arctan2(wx, wy)
    features["SignalWeighted_Azimuth_sin"] = np.sin(
        features["SignalWeighted_Azimuth_rad"]
    )
    features["SignalWeighted_Azimuth_cos"] = np.cos(
        features["SignalWeighted_Azimuth_rad"]
    )

    # Weighted Elevation: asin(z / r)
    wr = np.sqrt(wx**2 + wy**2 + wz**2)
    features["SignalWeighted_Elevation"] = np.degrees(np.arcsin(wz / (wr + 1e-9)))

    # Convert WLS ECEF to Lat/Lon
    # Drop epochs where WLS is missing
    features = features.dropna(subset=wls_cols)

    if features.empty:
        return None

    wls_lat, wls_lon = ecef_to_lla(
        features["WlsPositionXEcefMeters"].values,
        features["WlsPositionYEcefMeters"].values,
        features["WlsPositionZEcefMeters"].values,
    )

    features["WlsLatitudeDegrees"] = wls_lat
    features["WlsLongitudeDegrees"] = wls_lon

    # Prepare final dataframe
    # Ensure all feature columns exist and are filled
    for col in Config.FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0.0

    # Select features + metadata
    final_cols = Config.FEATURE_COLS + ["WlsLatitudeDegrees", "WlsLongitudeDegrees"]
    final_df = features[final_cols].copy()

    # Fill NaNs in features (e.g. std of 1 sample is NaN)
    final_df.fillna(0, inplace=True)

    # Add Index as column
    final_df["UnixTimeMillis"] = features.index
    final_df["drive_id"] = drive_id
    final_df["phone_name"] = phone_name

    # --- Targets (Training/Validation Only) ---
    if gt_df is not None:
        # Align GT to 1Hz Epochs
        gt_df = gt_df.copy()
        gt_df["Epoch"] = (
            np.round(gt_df["UnixTimeMillis"] / 1000.0).astype(np.int64) * 1000
        )

        # Merge features with GT
        # Use inner join: we only train on epochs where we have both GNSS and GT
        merged = pd.merge(
            final_df,
            gt_df[["Epoch", "LatitudeDegrees", "LongitudeDegrees"]],
            left_on="UnixTimeMillis",
            right_on="Epoch",
            how="inner",
        )

        # Compute Targets: Offset in Meters (East, North)
        # Target = GT - WLS
        dEast, dNorth = WGS84Utils.latlon_to_meters_diff(
            merged["WlsLatitudeDegrees"].values,
            merged["WlsLongitudeDegrees"].values,
            merged["LatitudeDegrees"].values,
            merged["LongitudeDegrees"].values,
        )

        merged["dEast"] = dEast
        merged["dNorth"] = dNorth

        return merged

    return final_df


def process_dataset(mode="train", load_cached_data=True):
    """
    Main function to process the dataset for the given mode.
    Handles caching and metadata iteration.
    """
    # Determine paths based on mode
    if mode == "train":
        meta_path = Config.TRAIN_META_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        meta_path = Config.VAL_META_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        meta_path = Config.TEST_META_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {mode} data from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Processing {mode} data from scratch...")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Identify unique trips (drive + phone)
    # For test, the metadata is already per-trip-timestamp, but we process per drive
    unique_trips = meta_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    results = []

    for _, row in unique_trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        # For train/val, we pass the ground truth subset to compute targets
        gt_subset = None
        if mode in ["train", "val"]:
            gt_subset = meta_df[
                (meta_df["drive_id"] == drive_id)
                & (meta_df["phone_name"] == phone_name)
            ].copy()

        processed_df = process_drive(drive_id, phone_name, gnss_path, gt_subset)

        if processed_df is not None and not processed_df.empty:
            results.append(processed_df)

    if not results:
        logger.warning(f"No data processed for {mode}!")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)

    # Save to cache
    logger.info(f"Saving {mode} data to {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
