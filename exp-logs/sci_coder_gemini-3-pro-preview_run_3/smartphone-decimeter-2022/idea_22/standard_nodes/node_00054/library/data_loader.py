import pandas as pd
import numpy as np
import os
from library.config import INPUT_DIR, SEED, WORKING_DIR
from library.utils import process_with_cache, load_metadata


def _aggregate_imu(imu_rel_path):
    """
    Loads and aggregates IMU data to 1Hz resolution.

    Args:
        imu_rel_path (str): Relative path to the IMU csv file.

    Returns:
        pd.DataFrame: Aggregated IMU features indexed by timestamp.
    """
    abs_path = os.path.join(INPUT_DIR, imu_rel_path)
    if not os.path.exists(abs_path):
        return pd.DataFrame()

    try:
        # Load IMU data
        df_imu = pd.read_csv(
            abs_path,
            usecols=[
                "utcTimeMillis",
                "MessageType",
                "MeasurementX",
                "MeasurementY",
                "MeasurementZ",
            ],
        )
    except ValueError:
        return pd.DataFrame()

    if df_imu.empty:
        return pd.DataFrame()

    # Round to nearest second (1000 ms) to align with GNSS epochs
    # GNSS epochs are typically at exact seconds.
    df_imu["UnixTimeMillis"] = (
        np.round(df_imu["utcTimeMillis"] / 1000.0).astype(np.int64) * 1000
    )

    # Aggregate by timestamp and sensor type
    agg_funcs = ["mean", "std"]
    grouped = df_imu.groupby(["UnixTimeMillis", "MessageType"])[
        ["MeasurementX", "MeasurementY", "MeasurementZ"]
    ].agg(agg_funcs)

    # Flatten MultiIndex columns
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Pivot to put sensor types in columns
    pivoted = grouped.pivot(index="UnixTimeMillis", columns="MessageType")

    # Flatten pivot MultiIndex columns (Metric_Type -> Type_Metric)
    # pivoted.columns is like (MeasurementX_mean, UncalAccel)
    new_cols = []
    for col in pivoted.columns.values:
        metric_name = col[0]
        sensor_type = col[1]
        new_cols.append(f"{sensor_type}_{metric_name}")

    pivoted.columns = new_cols
    return pivoted.reset_index()


def _process_drive(
    drive_id, phone_name, gnss_path, imu_path, gt_path=None, trip_id=None
):
    """
    Loads and merges data for a single drive-phone pair.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_path (str): Relative path to GNSS file.
        imu_path (str): Relative path to IMU file.
        gt_path (str, optional): Relative path to Ground Truth file.
        trip_id (str, optional): Unique trip identifier.

    Returns:
        pd.DataFrame: Merged dataframe for the trip.
    """
    # 1. Load GNSS
    gnss_abs_path = os.path.join(INPUT_DIR, gnss_path)
    if not os.path.exists(gnss_abs_path):
        return pd.DataFrame()

    df_gnss = pd.read_csv(gnss_abs_path)

    # Standardize Timestamp Column Name immediately
    # Cite debug_lesson_4: Synchronize Column References
    if "utcTimeMillis" in df_gnss.columns:
        df_gnss.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # 2. Load and Aggregate IMU
    df_imu_agg = _aggregate_imu(imu_path)

    # 3. Merge IMU onto GNSS
    if not df_imu_agg.empty:
        df_gnss = pd.merge(
            df_gnss,
            df_imu_agg,
            on="UnixTimeMillis",
            how="left",
        )

    # 4. Load Ground Truth (if provided)
    if gt_path:
        gt_abs_path = os.path.join(INPUT_DIR, gt_path)
        if os.path.exists(gt_abs_path):
            df_gt = pd.read_csv(gt_abs_path)

            # Merge GNSS with GT
            # Inner join ensures we only keep GNSS epochs where we have labels
            df_gnss = pd.merge(
                df_gnss,
                df_gt,
                on="UnixTimeMillis",
                how="inner",
                suffixes=("", "_gt"),
            )
        else:
            # If GT is missing but required (train/val), return empty to be safe
            return pd.DataFrame()

    # Add metadata identifiers
    df_gnss["drive_id"] = drive_id
    df_gnss["phone_name"] = phone_name

    # Explicitly attach tripId to propagate context. Cite debug_lesson_5.
    if trip_id:
        df_gnss["tripId"] = trip_id
    else:
        df_gnss["tripId"] = f"{drive_id}-{phone_name}"

    return df_gnss


def _compute_dataset(split, max_drives=None):
    """
    Internal worker function to build the dataset from metadata.

    Args:
        split (str): 'train', 'val', or 'test'.
        max_drives (int, optional): Limit number of drives processed.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    meta_df = load_metadata(split)

    # Identify unique trips. Metadata is row-per-epoch, so we drop duplicates to get file paths.
    cols_to_use = ["tripId", "drive_id", "phone_name", "gnss_path", "imu_path"]
    if "gt_path" in meta_df.columns:
        cols_to_use.append("gt_path")

    trips = meta_df[cols_to_use].drop_duplicates()

    if max_drives:
        # Cite debug_lesson_8: Satisfy CV Group Invariants.
        # Select by unique drive_id to ensure we get distinct groups for GroupKFold.
        unique_drives = trips["drive_id"].unique()[:max_drives]
        trips = trips[trips["drive_id"].isin(unique_drives)]

    all_data = []
    print(f"Processing {len(trips)} trips for split '{split}'...")

    for _, row in trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]
        imu_path = row["imu_path"]
        gt_path = row.get("gt_path")  # Returns None if key doesn't exist
        trip_id = row["tripId"]

        df_trip = _process_drive(
            drive_id, phone_name, gnss_path, imu_path, gt_path, trip_id=trip_id
        )

        if not df_trip.empty:
            all_data.append(df_trip)

    if not all_data:
        # Return empty dataframe with columns if possible, or just empty
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df


def load_dataset(split, load_cached_data=True, max_drives=None):
    """
    Public API to load the dataset for a specific split.
    Uses strict caching logic.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        max_drives (int): Optional limit on number of drives to process (for debugging).

    Returns:
        pd.DataFrame: The aligned dataset.
    """
    # Construct a unique cache filename based on parameters
    if max_drives is not None:
        cache_filename = f"dataset_{split}_{max_drives}.parquet"
    else:
        cache_filename = f"dataset_{split}.parquet"

    return process_with_cache(
        filename=cache_filename,
        processing_func=_compute_dataset,
        load_cached_data=load_cached_data,
        split=split,
        max_drives=max_drives,
    )
