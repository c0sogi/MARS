import os
import pandas as pd
import numpy as np
from library.config import Config


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata for a specific split (train, val, or test).

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
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)
    return df


def _get_cache_filename(drive_id: str, phone_name: str, data_type: str) -> str:
    """Generates a standardized cache filename."""
    # Sanitize strings to be safe for filenames
    safe_drive = drive_id.replace("/", "_").replace("\\", "_")
    safe_phone = phone_name.replace("/", "_").replace("\\", "_")
    return f"{safe_drive}_{safe_phone}_{data_type}.parquet"


def _load_or_cache_csv(
    relative_path: str, cache_name: str, load_cached_data: bool, time_col: str = None
) -> pd.DataFrame:
    """
    Helper function to load a CSV, optionally cache it as Parquet, and return the DataFrame.

    Args:
        relative_path: Path to CSV relative to Config.INPUT_DIR.
        cache_name: Filename for the cached parquet file.
        load_cached_data: Whether to attempt loading from cache.
        time_col: Name of the timestamp column to sort by.

    Returns:
        pd.DataFrame: Loaded data.
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "raw_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Warning: Failed to load cache {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source
    full_path = os.path.join(Config.INPUT_DIR, relative_path)
    if not os.path.exists(full_path):
        # Return empty dataframe if file missing (though metadata implies it exists)
        print(f"Warning: Source file not found: {full_path}")
        return pd.DataFrame()

    df = pd.read_csv(full_path)

    # 3. Process (Sort)
    if time_col and time_col in df.columns:
        df = df.sort_values(by=time_col).reset_index(drop=True)

    # 4. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to write cache {cache_path}: {e}")

    return df


def load_drive_data(
    drive_id: str,
    phone_name: str,
    gnss_path: str,
    imu_path: str,
    gt_path: str = None,
    load_cached_data: bool = True,
) -> dict:
    """
    Loads GNSS, IMU, and optionally Ground Truth data for a specific drive and phone.
    Handles caching to parquet for faster subsequent reads.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_path (str): Relative path to device_gnss.csv.
        imu_path (str): Relative path to device_imu.csv.
        gt_path (str, optional): Relative path to ground_truth.csv. Defaults to None.
        load_cached_data (bool, optional): Whether to use cached parquet files. Defaults to True.

    Returns:
        dict: Dictionary containing 'gnss', 'imu', and 'gt' (if provided) DataFrames.
    """

    # Define cache names
    # We prefix with train/test implicitly by using drive_id, but explicit types help
    # Note: drive_id usually contains the date, making it unique enough.

    gnss_cache_name = _get_cache_filename(drive_id, phone_name, "device_gnss")
    imu_cache_name = _get_cache_filename(drive_id, phone_name, "device_imu")

    # Load GNSS
    # Time column for GNSS is 'utcTimeMillis'
    gnss_df = _load_or_cache_csv(
        gnss_path, gnss_cache_name, load_cached_data, time_col="utcTimeMillis"
    )

    # Load IMU
    # Time column for IMU is 'utcTimeMillis'
    imu_df = _load_or_cache_csv(
        imu_path, imu_cache_name, load_cached_data, time_col="utcTimeMillis"
    )

    result = {"gnss": gnss_df, "imu": imu_df}

    # Load GT if provided
    if gt_path:
        gt_cache_name = _get_cache_filename(drive_id, phone_name, "ground_truth")
        # Time column for GT is 'UnixTimeMillis'
        gt_df = _load_or_cache_csv(
            gt_path, gt_cache_name, load_cached_data, time_col="UnixTimeMillis"
        )
        result["gt"] = gt_df
    else:
        result["gt"] = None

    return result
