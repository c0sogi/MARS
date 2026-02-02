import os
import numpy as np
import pandas as pd
import math
from library.config import Config
from library.utils import wgs84_to_cartesian

# -------------------------------------------------------------------------
# Coordinate Conversion Helpers
# -------------------------------------------------------------------------


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (WGS84).

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Latitude and Longitude in degrees, Altitude in meters.
    """
    # WGS84 ellipsoid constants
    a = Config.WGS84_A
    b = Config.WGS84_B
    e = np.sqrt(1 - (b**2 / a**2))
    ep = np.sqrt((a**2 / b**2) - 1)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)

    lat = np.arctan2(z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3)

    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def vectorize_ecef_to_lla(df, x_col, y_col, z_col, lat_col, lon_col):
    """
    Vectorized wrapper for ECEF to LLA conversion on a DataFrame.
    """
    # Using numpy arrays for speed
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    a = Config.WGS84_A
    b = Config.WGS84_B
    e2 = 1 - (b**2 / a**2)
    ep2 = (a**2 / b**2) - 1

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon_rad = np.arctan2(y, x)

    sin_th = np.sin(th)
    cos_th = np.cos(th)

    lat_rad = np.arctan2(z + ep2 * b * sin_th**3, p - e2 * a * cos_th**3)

    df[lat_col] = np.degrees(lat_rad)
    df[lon_col] = np.degrees(lon_rad)
    return df


# -------------------------------------------------------------------------
# Feature Engineering Logic
# -------------------------------------------------------------------------


def stratify_satellites(gnss_df):
    """
    Creates boolean masks for satellite strata.

    Strata:
    1. Global: All satellites (Implicit, no mask needed)
    2. High Precision: L5/E5a/B2a signals
    3. High Risk: Elevation < Threshold
    """
    # Stratum 2: High Precision
    # Check if SignalType is in the allowed L5 list
    # We assume SignalType column exists. If not, we might need to infer from frequency.
    # Based on dataset description, SignalType exists.
    if "SignalType" in gnss_df.columns:
        mask_l5 = gnss_df["SignalType"].isin(Config.L5_SIGNAL_TYPES)
    else:
        # Fallback if SignalType is missing (unlikely based on desc)
        mask_l5 = pd.Series(False, index=gnss_df.index)

    # Stratum 3: Low Elevation
    mask_low_elev = gnss_df["SvElevationDegrees"] < Config.LOW_ELEVATION_THRESHOLD

    return mask_l5, mask_low_elev


def compute_geometric_context(gnss_df):
    """
    Computes global geometric features per timestamp:
    - Signal Weighted Azimuth Centroid
    - Signal Count
    - Mean Pseudorange Uncertainty
    """
    # Weight by signal strength (convert dBHz to linear scale approximation)
    # w = 10^(Cn0 / 10)
    gnss_df["signal_weight"] = np.power(10, gnss_df["Cn0DbHz"] / 10.0)

    # Convert Azimuth to radians
    az_rad = np.deg2rad(gnss_df["SvAzimuthDegrees"])

    # Components
    gnss_df["az_x"] = gnss_df["signal_weight"] * np.cos(az_rad)
    gnss_df["az_y"] = gnss_df["signal_weight"] * np.sin(az_rad)

    # Aggregation
    grouped = gnss_df.groupby("UnixTimeMillis")

    # Sum components
    sums = grouped[["az_x", "az_y", "signal_weight"]].sum()

    # Centroid
    # Avoid division by zero
    sums["signal_weight"] = sums["signal_weight"].replace(0, 1.0)

    centroid_x = sums["az_x"] / sums["signal_weight"]
    centroid_y = sums["az_y"] / sums["signal_weight"]

    # Other globals
    sig_count = grouped.size()
    pr_unc = grouped["RawPseudorangeUncertaintyMeters"].mean()

    context_df = pd.DataFrame(
        {
            "SignalCount": sig_count,
            "RawPseudorangeUncertaintyMeters_mean": pr_unc,
            "AzimuthCentroid_X": centroid_x,
            "AzimuthCentroid_Y": centroid_y,
        }
    )

    return context_df


def aggregate_features(gnss_df):
    """
    Aggregates features for all strata and merges them.
    """
    # Identify Strata
    mask_l5, mask_low_elev = stratify_satellites(gnss_df)

    # Define aggregation dictionary
    # We aggregate Cn0DbHz and SvElevationDegrees
    aggs = {"Cn0DbHz": Config.STRATA_STATS, "SvElevationDegrees": Config.STRATA_STATS}

    # 1. Stratum 1: Global (All)
    s1 = gnss_df.groupby("UnixTimeMillis").agg(aggs)
    s1.columns = [f"S1_{c[0]}_{c[1]}" for c in s1.columns]

    # 2. Stratum 2: L5
    df_l5 = gnss_df[mask_l5]
    if not df_l5.empty:
        s2 = df_l5.groupby("UnixTimeMillis").agg(aggs)
        s2.columns = [f"S2_{c[0]}_{c[1]}" for c in s2.columns]
    else:
        # Create empty DataFrame with correct index and columns if no L5 signals
        cols = [f"S2_{k}_{stat}" for k in aggs.keys() for stat in aggs[k]]
        s2 = pd.DataFrame(0.0, index=s1.index, columns=cols)

    # 3. Stratum 3: Low Elevation
    df_low = gnss_df[mask_low_elev]
    if not df_low.empty:
        s3 = df_low.groupby("UnixTimeMillis").agg(aggs)
        s3.columns = [f"S3_{c[0]}_{c[1]}" for c in s3.columns]
    else:
        cols = [f"S3_{k}_{stat}" for k in aggs.keys() for stat in aggs[k]]
        s3 = pd.DataFrame(0.0, index=s1.index, columns=cols)

    # Merge Strata
    features = pd.concat([s1, s2, s3], axis=1)

    # Fill NaNs (e.g. if a timestamp has no L5 signals, merge produces NaNs)
    features = features.fillna(0.0)

    # Compute and Merge Global Context
    context = compute_geometric_context(gnss_df)
    features = features.join(context, how="left").fillna(0.0)

    return features


def process_drive(drive_id, phone_name, gnss_rel_path, gt_df=None):
    """
    Process raw data for a single drive/phone.

    Args:
        drive_id: ID of the drive.
        phone_name: Name of the phone.
        gnss_rel_path: Relative path to device_gnss.csv.
        gt_df: Ground truth DataFrame (optional).

    Returns:
        DataFrame containing features and (optional) targets aligned by timestamp.
    """
    gnss_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)

    if not os.path.exists(gnss_path):
        print(f"Warning: File not found {gnss_path}")
        return None

    # Load Raw GNSS
    # We only need specific columns to save memory
    use_cols = [
        "utcTimeMillis",
        "SignalType",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    # Note: 'MessageType' isn't always in the column list if we read specific cols,
    # but usually device_gnss.csv is filtered.
    # However, dataset desc says "Each row contains raw GNSS measurements... MessageType - Raw".
    # We should filter for MessageType == 'Raw' if we load it.

    try:
        # Load all columns first to safely filter
        df_raw = pd.read_csv(gnss_path)
        if "MessageType" in df_raw.columns:
            df_raw = df_raw[df_raw["MessageType"] == "Raw"].copy()
    except Exception as e:
        print(f"Error reading {gnss_path}: {e}")
        return None

    # Time Quantization
    # Round utcTimeMillis to nearest second (1000ms) to align with GT
    df_raw["UnixTimeMillis"] = (
        np.round(df_raw["utcTimeMillis"] / 1000.0) * 1000
    ).astype(np.int64)

    # Extract WLS Baseline (One per timestamp)
    # Use mean to handle missing values (NaNs) in some rows for the same epoch (Cite debug_lesson_6)
    wls_df = df_raw.groupby("UnixTimeMillis")[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].mean()

    # Drop rows where WLS is missing (prevents NaN targets)
    wls_df = wls_df.dropna()

    # Convert WLS ECEF to LLA
    wls_df = vectorize_ecef_to_lla(
        wls_df,
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "WlsLat",
        "WlsLon",
    )

    # Aggregate Features
    features_df = aggregate_features(df_raw)

    # Merge Features with WLS Baseline
    # Inner join ensures we only keep timestamps where we have both features and baseline
    processed_df = features_df.join(wls_df[["WlsLat", "WlsLon"]], how="inner")

    # Add Metadata
    processed_df["drive_id"] = drive_id
    processed_df["phone_name"] = phone_name

    # Handle Ground Truth (Targets)
    if gt_df is not None:
        # GT timestamps are already UnixTimeMillis
        # Filter GT for this drive/phone
        subset_gt = gt_df[
            (gt_df["drive_id"] == drive_id) & (gt_df["phone_name"] == phone_name)
        ].copy()

        # Round GT timestamps to nearest second to align with GNSS data (Cite debug_lesson_4)
        subset_gt["UnixTimeMillis"] = (
            np.round(subset_gt["UnixTimeMillis"] / 1000.0) * 1000
        ).astype(np.int64)

        subset_gt = subset_gt.set_index("UnixTimeMillis")
        subset_gt = subset_gt[["LatitudeDegrees", "LongitudeDegrees"]]
        subset_gt.columns = ["GtLat", "GtLon"]

        # Merge with processed features
        # Use inner join to only keep rows with labels
        processed_df = processed_df.join(subset_gt, how="inner")

        # Compute Targets (Cartesian Offsets in Meters)
        # Target = GT - WLS
        d_east, d_north = wgs84_to_cartesian(
            processed_df["GtLat"].values,
            processed_df["GtLon"].values,
            processed_df["WlsLat"].values,
            processed_df["WlsLon"].values,
        )

        processed_df["dEast"] = d_east
        processed_df["dNorth"] = d_north

    return processed_df.reset_index()


# -------------------------------------------------------------------------
# Main Processing Pipeline
# -------------------------------------------------------------------------


def process_dataset(metadata_path, load_cached_data=True, split_name="train"):
    """
    Main function to process a dataset defined by a metadata file.
    Handles caching.

    Args:
        metadata_path: Path to the metadata CSV.
        load_cached_data: If True, attempts to load from parquet cache.
        split_name: Name of the split ('train', 'val', 'test') for cache naming.

    Returns:
        pd.DataFrame: Processed dataset ready for the model.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        try:
            df = pd.read_parquet(cache_file)

            # Validate Schema (Cite debug_lesson_13)
            expected_cols = []
            # Reconstruct expected feature columns to validate cache
            for i in [1, 2, 3]:
                for var in ["Cn0DbHz", "SvElevationDegrees"]:
                    for stat in Config.STRATA_STATS:
                        expected_cols.append(f"S{i}_{var}_{stat}")
            expected_cols.extend(Config.GLOBAL_FEATURES)

            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(
                    f"Cache invalid: Missing {len(missing)} columns (e.g., {missing[:3]}). Reprocessing..."
                )
            else:
                return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # If debugging, sample drives
    if Config.DEBUG:
        drives = meta_df["drive_id"].unique()
        sampled_drives = drives[: Config.DEBUG_DRIVE_COUNT]
        meta_df = meta_df[meta_df["drive_id"].isin(sampled_drives)]
        print(f"DEBUG MODE: Sampling {len(sampled_drives)} drives.")

    # Group by drive and phone to process sequentially
    # We need unique (drive_id, phone_name) pairs
    # In train/val metadata, rows are samples. In test metadata, rows are samples.
    # We extract unique pairs to process the raw files once per phone run.
    unique_runs = meta_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    results = []

    for _, row in unique_runs.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        # Pass GT dataframe if it's train/val
        # The meta_df itself contains GT columns if they exist
        if "LatitudeDegrees" in meta_df.columns:
            processed_run = process_drive(
                drive_id, phone_name, gnss_path, gt_df=meta_df
            )
        else:
            processed_run = process_drive(drive_id, phone_name, gnss_path, gt_df=None)

        if processed_run is not None and not processed_run.empty:
            results.append(processed_run)

    if not results:
        raise ValueError("No data processed!")

    final_df = pd.concat(results, ignore_index=True)

    # 3. Save Cache
    print(f"Saving {split_name} data to {cache_file}...")
    final_df.to_parquet(cache_file, index=False)

    return final_df
