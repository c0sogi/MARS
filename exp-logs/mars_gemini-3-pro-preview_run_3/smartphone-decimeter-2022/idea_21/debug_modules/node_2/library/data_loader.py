import os
import pandas as pd
import numpy as np
from library.config import INPUT_DIR, WORKING_DIR


def align_timestamps(gnss_df, gt_df):
    """
    Synchronize GNSS measurements with the 1Hz ground truth clock.
    Uses nearest neighbor alignment within a tolerance of 1 second.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data containing 'utcTimeMillis'.
        gt_df (pd.DataFrame): Ground truth/Target data containing 'UnixTimeMillis'.

    Returns:
        pd.DataFrame: GNSS data merged with Ground Truth, aligned by timestamp.
    """
    # Ensure timestamps are sorted for merge_asof
    gnss_df = gnss_df.sort_values("utcTimeMillis")
    gt_df = gt_df.sort_values("UnixTimeMillis")

    # 1. Identify unique epochs in GNSS to map to GT timestamps
    gnss_epochs = (
        gnss_df[["utcTimeMillis"]].drop_duplicates().sort_values("utcTimeMillis")
    )

    # 2. Find nearest GNSS epoch for each GT timestamp
    # tolerance=1000ms ensures we don't match signals too far away (e.g. gaps)
    alignment = pd.merge_asof(
        gt_df[["UnixTimeMillis"]],
        gnss_epochs,
        left_on="UnixTimeMillis",
        right_on="utcTimeMillis",
        direction="nearest",
        tolerance=1000,
    )

    # Filter out GT timestamps that didn't find a matching GNSS epoch
    valid_alignment = alignment.dropna(subset=["utcTimeMillis"])

    # 3. Merge the full GNSS data onto this alignment map
    # This replicates the target timestamp for each satellite signal in that epoch
    aligned_gnss = pd.merge(valid_alignment, gnss_df, on="utcTimeMillis", how="inner")

    # 4. Merge the full GT data (targets/auxiliary info) onto the aligned GNSS
    final_df = pd.merge(aligned_gnss, gt_df, on="UnixTimeMillis", how="left")

    return final_df


def load_drive_data(
    drive_id, phone_name, metadata_df, input_dir=INPUT_DIR, load_cached_data=True
):
    """
    Load and align data for a specific drive and phone.

    Args:
        drive_id (str): Drive ID.
        phone_name (str): Phone name.
        metadata_df (pd.DataFrame): Metadata containing target timestamps and file paths.
                                    Must contain rows for the specific drive/phone.
        input_dir (str): Root directory for input data.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (aligned_gnss_df, raw_imu_df)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache path for the aligned GNSS data
    cache_file = f"aligned_{drive_id}_{phone_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_file)

    # Extract file paths from metadata (take first row as paths are constant for the drive)
    drive_meta = metadata_df[
        (metadata_df["drive_id"] == drive_id)
        & (metadata_df["phone_name"] == phone_name)
    ]

    if drive_meta.empty:
        raise ValueError(f"No metadata found for {drive_id} {phone_name}")

    first_row = drive_meta.iloc[0]

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load aligned GNSS from cache
            aligned_gnss_df = pd.read_parquet(cache_path)

            # Load Raw IMU (we don't cache raw IMU as it's just a direct read)
            imu_rel_path = first_row["imu_path"]
            imu_path = os.path.join(input_dir, imu_rel_path)
            if os.path.exists(imu_path):
                raw_imu_df = pd.read_csv(imu_path)
            else:
                raw_imu_df = pd.DataFrame()

            return aligned_gnss_df, raw_imu_df
        except Exception as e:
            print(
                f"Warning: Failed to load cache for {drive_id}-{phone_name}, recomputing. Error: {e}"
            )

    # --- Compute from scratch ---

    # 1. Load Raw GNSS
    gnss_path = os.path.join(input_dir, first_row["gnss_path"])
    if os.path.exists(gnss_path):
        gnss_df = pd.read_csv(gnss_path)
    else:
        raise FileNotFoundError(f"GNSS file not found: {gnss_path}")

    # 2. Load Raw IMU
    imu_path = os.path.join(input_dir, first_row["imu_path"])
    if os.path.exists(imu_path):
        raw_imu_df = pd.read_csv(imu_path)
    else:
        raw_imu_df = pd.DataFrame()

    # 3. Load Ground Truth / Targets
    # If 'gt_path' exists in metadata (Training set), load the full GT file
    # Otherwise (Test set), use the metadata itself which contains the target timestamps
    if "gt_path" in first_row and pd.notna(first_row["gt_path"]):
        gt_file_path = os.path.join(input_dir, first_row["gt_path"])
        if os.path.exists(gt_file_path):
            gt_df = pd.read_csv(gt_file_path)
        else:
            gt_df = drive_meta
    else:
        gt_df = drive_meta

    # 4. Align GNSS to GT timestamps
    aligned_gnss_df = align_timestamps(gnss_df, gt_df)

    # Add context identifiers (Cite debug_lesson_15)
    aligned_gnss_df["drive_id"] = drive_id
    aligned_gnss_df["phone_name"] = phone_name

    # 5. Save to Cache
    try:
        aligned_gnss_df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache for {drive_id}-{phone_name}: {e}")

    return aligned_gnss_df, raw_imu_df
