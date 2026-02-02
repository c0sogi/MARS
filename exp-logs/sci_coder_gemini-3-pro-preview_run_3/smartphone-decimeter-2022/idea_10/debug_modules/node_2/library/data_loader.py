import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    GNSS_COLS,
    IMU_COLS,
)


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df


def _load_raw_data(
    metadata_df,
    path_col,
    use_cols,
    cache_name,
    load_cached_data=True,
    max_files=None,
    rename_time_col=True,
):
    """
    Helper function to load raw data files referenced in the metadata.
    Implements caching logic.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing file paths.
        path_col (str): Column name in metadata_df containing relative paths.
        use_cols (list): List of columns to read from the CSVs.
        cache_name (str): Name of the cache file (e.g., 'train_gnss').
        load_cached_data (bool): Whether to try loading from cache.
        max_files (int, optional): Limit number of files to process (for debugging).
        rename_time_col (bool): If True, renames 'utcTimeMillis' to 'UnixTimeMillis'.

    Returns:
        pd.DataFrame: Concatenated raw data.
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing raw data for {cache_name}...")

    # Get unique file paths and associated drive/phone info
    # We need drive_id and phone_name to merge back later if necessary,
    # although the raw files are usually merged by time.
    # Adding drive_id/phone_name to raw rows is safer.
    meta_cols = [path_col, "drive_id", "phone_name"]
    if "tripId" in metadata_df.columns:
        meta_cols.append("tripId")

    unique_files = metadata_df[meta_cols].drop_duplicates()

    if max_files:
        unique_files = unique_files.head(max_files)
        print(f"Limiting to {max_files} files.")

    data_frames = []

    for _, row in unique_files.iterrows():
        rel_path = row[path_col]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            # This might happen if metadata points to a missing file (unlikely with generated metadata)
            continue

        try:
            # Read only necessary columns
            # Note: use_cols must exist in the file.
            # device_gnss.csv and device_imu.csv usually have standard headers.
            df = pd.read_csv(full_path, usecols=lambda c: c in use_cols)

            # Add identifiers
            df["drive_id"] = drive_id
            df["phone_name"] = phone_name
            if "tripId" in row:
                df["tripId"] = row["tripId"]

            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")

    if not data_frames:
        print("Warning: No data loaded.")
        return pd.DataFrame()

    result_df = pd.concat(data_frames, ignore_index=True)

    # Rename time column to match metadata/ground_truth standard
    if rename_time_col and "utcTimeMillis" in result_df.columns:
        result_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    # 3. Save to cache
    print(f"Saving cache to {cache_path}...")
    result_df.to_parquet(cache_path, index=False)

    return result_df


def load_gnss_raw(
    metadata_df, split_name="train", load_cached_data=True, max_files=None
):
    """
    Loads raw GNSS data for the given metadata.

    Args:
        metadata_df (pd.DataFrame): Metadata dataframe.
        split_name (str): 'train', 'val', or 'test' (used for cache naming).
        load_cached_data (bool): Use cache if available.
        max_files (int): Limit files for debugging.

    Returns:
        pd.DataFrame: Aggregated GNSS data.
    """
    return _load_raw_data(
        metadata_df=metadata_df,
        path_col="gnss_path",
        use_cols=GNSS_COLS,
        cache_name=f"{split_name}_gnss_raw",
        load_cached_data=load_cached_data,
        max_files=max_files,
        rename_time_col=True,
    )


def load_imu_raw(
    metadata_df, split_name="train", load_cached_data=True, max_files=None
):
    """
    Loads raw IMU data for the given metadata.

    Args:
        metadata_df (pd.DataFrame): Metadata dataframe.
        split_name (str): 'train', 'val', or 'test' (used for cache naming).
        load_cached_data (bool): Use cache if available.
        max_files (int): Limit files for debugging.

    Returns:
        pd.DataFrame: Aggregated IMU data.
    """
    return _load_raw_data(
        metadata_df=metadata_df,
        path_col="imu_path",
        use_cols=IMU_COLS,
        cache_name=f"{split_name}_imu_raw",
        load_cached_data=load_cached_data,
        max_files=max_files,
        rename_time_col=True,
    )


def load_ground_truth(
    metadata_df, split_name="train", load_cached_data=True, max_files=None
):
    """
    Loads full ground truth data (including Speed, Altitude, etc.) for the given metadata.
    Only applicable for train/val splits where 'gt_path' exists.

    Args:
        metadata_df (pd.DataFrame): Metadata dataframe.
        split_name (str): 'train' or 'val'.
        load_cached_data (bool): Use cache if available.
        max_files (int): Limit files for debugging.

    Returns:
        pd.DataFrame: Aggregated Ground Truth data.
    """
    if "gt_path" not in metadata_df.columns:
        raise ValueError(
            "Metadata does not contain 'gt_path'. Cannot load ground truth."
        )

    # All columns in GT are potentially useful for analysis, but we can filter if needed.
    # We'll load all for now as GT files are small compared to GNSS.
    # Common GT cols: LatitudeDegrees, LongitudeDegrees, AltitudeMeters, SpeedMps, AccuracyMeters, BearingDegrees, UnixTimeMillis

    # We define a broad list of columns to keep if they exist
    gt_cols_to_load = [
        "LatitudeDegrees",
        "LongitudeDegrees",
        "AltitudeMeters",
        "SpeedMps",
        "AccuracyMeters",
        "BearingDegrees",
        "UnixTimeMillis",
    ]

    return _load_raw_data(
        metadata_df=metadata_df,
        path_col="gt_path",
        use_cols=gt_cols_to_load,
        cache_name=f"{split_name}_ground_truth",
        load_cached_data=load_cached_data,
        max_files=max_files,
        rename_time_col=False,  # GT already has UnixTimeMillis
    )
