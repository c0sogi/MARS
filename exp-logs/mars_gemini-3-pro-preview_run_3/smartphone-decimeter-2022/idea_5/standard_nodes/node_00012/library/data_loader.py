import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata file for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def load_drive_data(drive_id: str, phone_name: str, metadata_df: pd.DataFrame) -> tuple:
    """
    Loads raw data for a specific drive and phone.
    Merges Ground Truth with GNSS data if Ground Truth is available.

    Args:
        drive_id (str): The drive identifier.
        phone_name (str): The phone model name.
        metadata_df (pd.DataFrame): The metadata dataframe containing file paths.

    Returns:
        tuple: (gnss_df, imu_df)
            gnss_df (pd.DataFrame): GNSS data (merged with GT if available).
            imu_df (pd.DataFrame): IMU data.
    """
    # Filter metadata for this drive/phone
    # We take the first row to get the paths (paths are constant for the drive/phone)
    subset = metadata_df[
        (metadata_df["drive_id"] == drive_id)
        & (metadata_df["phone_name"] == phone_name)
    ]

    if subset.empty:
        raise ValueError(
            f"No metadata found for drive {drive_id} and phone {phone_name}"
        )

    row = subset.iloc[0]

    # Construct absolute paths
    gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])
    imu_path = os.path.join(Config.INPUT_DIR, row["imu_path"])

    # Load GNSS
    if not os.path.exists(gnss_path):
        raise FileNotFoundError(f"GNSS file not found: {gnss_path}")
    gnss_df = pd.read_csv(gnss_path)

    # Load IMU
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU file not found: {imu_path}")
    imu_df = pd.read_csv(imu_path)

    # Load Ground Truth if available and path exists
    # Note: Test set metadata usually won't have 'gt_path' or it will be null
    if "gt_path" in row and pd.notna(row["gt_path"]):
        gt_path = os.path.join(Config.INPUT_DIR, row["gt_path"])
        if os.path.exists(gt_path):
            gt_df = pd.read_csv(gt_path)

            # Align timestamps: GT uses UnixTimeMillis, GNSS uses utcTimeMillis
            # We rename GT column to match GNSS for merging
            if "UnixTimeMillis" in gt_df.columns:
                gt_df = gt_df.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

            # Keep only relevant GT columns to avoid clutter
            cols_to_keep = [
                "utcTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
                "SpeedMps",
                "AccuracyMeters",
                "BearingDegrees",
            ]
            gt_df = gt_df[[c for c in cols_to_keep if c in gt_df.columns]]

            # Merge GT onto GNSS
            # We use inner join to keep only labeled epochs for training/validation
            # GNSS data has multiple rows per epoch (one per satellite)
            gnss_df = pd.merge(gnss_df, gt_df, on="utcTimeMillis", how="inner")

    # Add drive_id and phone_name columns for reference/grouping
    gnss_df["drive_id"] = drive_id
    gnss_df["phone_name"] = phone_name
    imu_df["drive_id"] = drive_id
    imu_df["phone_name"] = phone_name

    return gnss_df, imu_df


def load_split_data(split: str, load_cached_data: bool = True) -> tuple:
    """
    Loads all data for a specific split (train/val/test).
    Implements caching to Parquet to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (gnss_all, imu_all)
    """
    cache_dir = Config.WORKING_DIR
    gnss_cache_path = os.path.join(cache_dir, f"{split}_gnss_raw.parquet")
    imu_cache_path = os.path.join(cache_dir, f"{split}_imu_raw.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(gnss_cache_path) and os.path.exists(imu_cache_path):
            logger.info(f"Loading cached {split} data from {cache_dir}...")
            try:
                gnss_all = pd.read_parquet(gnss_cache_path)
                imu_all = pd.read_parquet(imu_cache_path)
                return gnss_all, imu_all
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info(f"Cache not found for {split}. Computing from scratch...")

    # 2. Compute from scratch
    meta_df = load_metadata(split)

    # Get unique drive/phone combinations
    # metadata_df has one row per timestamp, so we drop duplicates to iterate over drives
    unique_trips = meta_df[["drive_id", "phone_name"]].drop_duplicates()

    gnss_list = []
    imu_list = []

    logger.info(f"Loading {len(unique_trips)} trips for {split} split...")

    for _, row in unique_trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        try:
            g_df, i_df = load_drive_data(drive_id, phone_name, meta_df)
            gnss_list.append(g_df)
            imu_list.append(i_df)
        except Exception as e:
            logger.error(f"Error loading drive {drive_id} {phone_name}: {e}")
            continue

    if not gnss_list:
        raise ValueError(f"No data loaded for split {split}")

    gnss_all = pd.concat(gnss_list, ignore_index=True)
    imu_all = pd.concat(imu_list, ignore_index=True)

    # 3. Save to cache
    logger.info(f"Saving {split} data to cache...")
    try:
        gnss_all.to_parquet(gnss_cache_path, index=False)
        imu_all.to_parquet(imu_cache_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    return gnss_all, imu_all
