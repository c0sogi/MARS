import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEBUG,
    DEBUG_DRIVE_COUNT,
    SEED,
)


def load_metadata(split: str) -> pd.DataFrame:
    """
    Load metadata for a specific split from the generated CSV files.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file not found at {path}. Please ensure metadata generation was successful."
        )

    return pd.read_csv(path)


def load_sensor_data(
    drive_id: str, phone_name: str, gnss_rel_path: str, imu_rel_path: str
):
    """
    Load raw GNSS and IMU CSV files for a specific drive and phone.

    Args:
        drive_id (str): The drive identifier.
        phone_name (str): The phone model name.
        gnss_rel_path (str): Relative path to device_gnss.csv.
        imu_rel_path (str): Relative path to device_imu.csv.

    Returns:
        tuple: (gnss_df, imu_df) as pandas DataFrames.
    """
    gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
    imu_path = os.path.join(INPUT_DIR, imu_rel_path)

    # Load GNSS
    if os.path.exists(gnss_path):
        gnss_df = pd.read_csv(gnss_path)
        gnss_df["drive_id"] = drive_id
        gnss_df["phone_name"] = phone_name
    else:
        print(f"Warning: GNSS file not found: {gnss_path}")
        gnss_df = pd.DataFrame()

    # Load IMU
    if os.path.exists(imu_path):
        imu_df = pd.read_csv(imu_path)
        imu_df["drive_id"] = drive_id
        imu_df["phone_name"] = phone_name
    else:
        print(f"Warning: IMU file not found: {imu_path}")
        imu_df = pd.DataFrame()

    return gnss_df, imu_df


def _get_data(split: str, load_cached_data: bool = True):
    """
    Internal helper to load, aggregate, and cache data for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempt to load from parquet cache.

    Returns:
        tuple: (gnss_df, imu_df, meta_df)
    """
    # Define cache paths
    cache_gnss_path = os.path.join(WORKING_DIR, f"{split}_gnss.parquet")
    cache_imu_path = os.path.join(WORKING_DIR, f"{split}_imu.parquet")
    cache_meta_path = os.path.join(WORKING_DIR, f"{split}_meta.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_gnss_path)
            and os.path.exists(cache_imu_path)
            and os.path.exists(cache_meta_path)
        ):
            print(f"[{split.upper()}] Loading data from cache...")
            try:
                gnss_df = pd.read_parquet(cache_gnss_path)
                imu_df = pd.read_parquet(cache_imu_path)
                meta_df = pd.read_parquet(cache_meta_path)
                return gnss_df, imu_df, meta_df
            except Exception as e:
                print(f"[{split.upper()}] Failed to load cache: {e}. Recomputing...")
        else:
            print(f"[{split.upper()}] Cache not found. Computing from raw files...")
    else:
        print(f"[{split.upper()}] Force recomputing data...")

    # 2. Load Metadata
    meta_df = load_metadata(split)

    # 3. Handle DEBUG mode (subsample drives)
    if DEBUG:
        unique_drives = meta_df["drive_id"].unique()
        if len(unique_drives) > DEBUG_DRIVE_COUNT:
            rng = np.random.RandomState(SEED)
            sampled_drives = rng.choice(
                unique_drives, size=DEBUG_DRIVE_COUNT, replace=False
            )
            print(f"[{split.upper()}] DEBUG: Sampling drives: {sampled_drives}")
            meta_df = meta_df[meta_df["drive_id"].isin(sampled_drives)].copy()

    # 4. Aggregate Sensor Data
    # Get unique drive-phone pairs to avoid loading the same file multiple times
    # (Metadata has one row per timestamp, but files are per drive-phone)
    sensor_files_map = meta_df[
        ["drive_id", "phone_name", "gnss_path", "imu_path"]
    ].drop_duplicates()

    gnss_list = []
    imu_list = []

    print(
        f"[{split.upper()}] Loading sensor files for {len(sensor_files_map)} drive-phone pairs..."
    )

    for _, row in sensor_files_map.iterrows():
        g_df, i_df = load_sensor_data(
            row["drive_id"], row["phone_name"], row["gnss_path"], row["imu_path"]
        )
        if not g_df.empty:
            gnss_list.append(g_df)
        if not i_df.empty:
            imu_list.append(i_df)

    if gnss_list:
        gnss_df = pd.concat(gnss_list, ignore_index=True)
    else:
        gnss_df = pd.DataFrame()

    if imu_list:
        imu_df = pd.concat(imu_list, ignore_index=True)
    else:
        imu_df = pd.DataFrame()

    # 5. Save to cache
    print(f"[{split.upper()}] Saving data to cache...")
    gnss_df.to_parquet(cache_gnss_path, index=False)
    imu_df.to_parquet(cache_imu_path, index=False)
    meta_df.to_parquet(cache_meta_path, index=False)

    return gnss_df, imu_df, meta_df


def get_train_data(load_cached_data: bool = True):
    """
    Get aggregated training data.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (gnss_df, imu_df, meta_df)
    """
    return _get_data("train", load_cached_data)


def get_val_data(load_cached_data: bool = True):
    """
    Get aggregated validation data.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (gnss_df, imu_df, meta_df)
    """
    return _get_data("val", load_cached_data)


def get_test_data(load_cached_data: bool = True):
    """
    Get aggregated test data.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (gnss_df, imu_df, meta_df)
    """
    return _get_data("test", load_cached_data)
