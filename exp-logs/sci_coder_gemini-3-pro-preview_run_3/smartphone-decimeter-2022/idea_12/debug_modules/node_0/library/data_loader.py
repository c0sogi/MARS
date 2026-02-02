import os
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_12"
METADATA_DIR = "./metadata"


def load_metadata(split):
    """
    Loads the metadata CSV for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def load_drive_data(
    drive_id, phone_name, gnss_rel_path, imu_rel_path, gt_rel_path=None
):
    """
    Loads raw sensor data for a single drive from CSV files.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_rel_path (str): Relative path to device_gnss.csv.
        imu_rel_path (str): Relative path to device_imu.csv.
        gt_rel_path (str, optional): Relative path to ground_truth.csv.

    Returns:
        tuple: (gnss_df, imu_df, gt_df)
    """
    gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
    imu_path = os.path.join(INPUT_DIR, imu_rel_path)

    # GNSS Columns relevant for physics-based feature engineering
    gnss_cols = [
        "utcTimeMillis",
        "Svid",
        "ConstellationType",
        "SignalType",
        "Cn0DbHz",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "RawPseudorangeMeters",
        "BiasNanos",
        "DriftNanosPerSecond",
    ]

    # Check header to avoid errors if columns are missing in some files
    try:
        header = pd.read_csv(gnss_path, nrows=0).columns
        use_cols = [c for c in gnss_cols if c in header]
        gnss_df = pd.read_csv(gnss_path, usecols=use_cols)
    except Exception as e:
        print(f"Error loading GNSS for {drive_id}-{phone_name}: {e}")
        gnss_df = pd.DataFrame(columns=["utcTimeMillis"])  # Empty fallback

    # IMU Columns
    imu_cols = [
        "utcTimeMillis",
        "MessageType",
        "MeasurementX",
        "MeasurementY",
        "MeasurementZ",
    ]
    try:
        imu_df = pd.read_csv(imu_path, usecols=imu_cols)
    except Exception as e:
        print(f"Error loading IMU for {drive_id}-{phone_name}: {e}")
        imu_df = pd.DataFrame(columns=["utcTimeMillis", "MessageType"])

    # Ground Truth
    gt_df = None
    if gt_rel_path:
        gt_full_path = os.path.join(INPUT_DIR, gt_rel_path)
        if os.path.exists(gt_full_path):
            gt_df = pd.read_csv(gt_full_path)

    return gnss_df, imu_df, gt_df


def merge_sensors(gnss_df, imu_df, target_timestamps):
    """
    Aligns GNSS and IMU data to the target timestamps.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data.
        imu_df (pd.DataFrame): Raw IMU data.
        target_timestamps (pd.Series): Target UnixTimeMillis.

    Returns:
        tuple: (aligned_gnss, aligned_imu)
    """
    # Unique sorted targets
    targets = np.sort(np.unique(target_timestamps))
    target_df = pd.DataFrame({"UnixTimeMillis": targets})

    # --- Align GNSS ---
    # GNSS is typically 1Hz. We filter to rows exactly matching target times.
    if "utcTimeMillis" in gnss_df.columns:
        gnss_df = gnss_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

    # Filter GNSS
    aligned_gnss = gnss_df[gnss_df["UnixTimeMillis"].isin(targets)].copy()

    # --- Align IMU ---
    # IMU is high frequency. We aggregate to the nearest second.
    # We round IMU timestamps to the nearest second (1000ms) to align with targets.
    if not imu_df.empty:
        imu_df["UnixTimeMillis"] = np.round(imu_df["utcTimeMillis"] / 1000.0) * 1000.0
        imu_df["UnixTimeMillis"] = imu_df["UnixTimeMillis"].astype(np.int64)

        # Filter to relevant timeframe to speed up groupby
        t_min = targets.min()
        t_max = targets.max()
        imu_subset = imu_df[
            (imu_df["UnixTimeMillis"] >= t_min) & (imu_df["UnixTimeMillis"] <= t_max)
        ]

        # Aggregate mean per timestamp and message type
        imu_agg = (
            imu_subset.groupby(["UnixTimeMillis", "MessageType"])[
                ["MeasurementX", "MeasurementY", "MeasurementZ"]
            ]
            .mean()
            .unstack()
        )

        # Flatten MultiIndex columns (e.g., MeasurementX_UncalAccel)
        imu_agg.columns = [f"{c[1]}_{c[0]}" for c in imu_agg.columns]
        imu_agg = imu_agg.reset_index()

        # Merge with all targets to ensure we have rows for every target (fill missing with NaN)
        aligned_imu = pd.merge(target_df, imu_agg, on="UnixTimeMillis", how="left")
    else:
        # Return empty dataframe with target timestamps if no IMU
        aligned_imu = target_df.copy()

    return aligned_gnss, aligned_imu


def load_dataset(split, load_cached_data=True):
    """
    Loads the full dataset for a given split, using caching to Parquet.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (final_gnss, final_imu, final_gt)
               final_gt contains target labels and timestamps.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    gnss_cache_path = os.path.join(CACHE_DIR, f"{split}_gnss.parquet")
    imu_cache_path = os.path.join(CACHE_DIR, f"{split}_imu.parquet")
    gt_cache_path = os.path.join(CACHE_DIR, f"{split}_gt.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(gnss_cache_path)
            and os.path.exists(imu_cache_path)
            and os.path.exists(gt_cache_path)
        ):
            print(f"[{split.upper()}] Loading cached data from {CACHE_DIR}...")
            try:
                gnss_df = pd.read_parquet(gnss_cache_path)
                imu_df = pd.read_parquet(imu_cache_path)
                gt_df = pd.read_parquet(gt_cache_path)
                return gnss_df, imu_df, gt_df
            except Exception as e:
                print(f"[{split.upper()}] Cache load failed ({e}). Reprocessing...")

    # 2. Process from Scratch
    print(f"[{split.upper()}] Processing data from scratch...")
    meta_df = load_metadata(split)

    all_gnss = []
    all_imu = []
    all_gt = []

    # Group by tripId to avoid redundant file IO (metadata is row-wise per timestamp)
    unique_trips = meta_df[
        ["tripId", "drive_id", "phone_name", "gnss_path", "imu_path"]
    ].drop_duplicates()

    print(f"[{split.upper()}] Processing {len(unique_trips)} unique trips...")

    has_gt_path = "gt_path" in meta_df.columns

    for i, row in unique_trips.iterrows():
        trip_id = row["tripId"]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Determine Ground Truth Path
        gt_rel_path = None
        if has_gt_path:
            # Get path from metadata (all rows for trip have same path)
            gt_rel_path = meta_df[meta_df["tripId"] == trip_id]["gt_path"].iloc[0]

        # Load Raw Data
        gnss_raw, imu_raw, gt_raw = load_drive_data(
            drive_id, phone_name, row["gnss_path"], row["imu_path"], gt_rel_path
        )

        # Determine Target Timestamps
        if gt_raw is not None:
            target_timestamps = gt_raw["UnixTimeMillis"]
            gt_subset = gt_raw.copy()
            gt_subset["tripId"] = trip_id
        else:
            # For test set, get timestamps from metadata
            target_timestamps = meta_df[meta_df["tripId"] == trip_id]["UnixTimeMillis"]
            # Create placeholder GT
            gt_subset = pd.DataFrame(
                {"tripId": trip_id, "UnixTimeMillis": target_timestamps}
            )

        # Align Sensors
        gnss_aligned, imu_aligned = merge_sensors(gnss_raw, imu_raw, target_timestamps)

        # Add Trip Context
        gnss_aligned["tripId"] = trip_id
        imu_aligned["tripId"] = trip_id

        all_gnss.append(gnss_aligned)
        all_imu.append(imu_aligned)
        all_gt.append(gt_subset)

    # Concatenate All
    final_gnss = pd.concat(all_gnss, ignore_index=True)
    final_imu = pd.concat(all_imu, ignore_index=True)
    final_gt = pd.concat(all_gt, ignore_index=True)

    # Save to Cache
    print(f"[{split.upper()}] Saving to cache...")
    final_gnss.to_parquet(gnss_cache_path, index=False)
    final_imu.to_parquet(imu_cache_path, index=False)
    final_gt.to_parquet(gt_cache_path, index=False)

    return final_gnss, final_imu, final_gt
