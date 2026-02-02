import os
import pandas as pd
import numpy as np
from library.data_loader import load_metadata, load_drive_data
from library.gnss_physics import generate_kinematic_features

# Constants
CACHE_DIR = "./working/idea_18"
os.makedirs(CACHE_DIR, exist_ok=True)


def process_drive(drive_id, phone_name, gnss_path, gt_path=None, is_train=True):
    """
    Processes a single drive: loads aligned data, generates dual-projection features,
    and aligns with ground truth targets if available.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model identifier.
        gnss_path (str): Relative path to GNSS log.
        gt_path (str, optional): Relative path to Ground Truth log.
        is_train (bool): True if processing training/validation data.

    Returns:
        pd.DataFrame: Feature matrix for the drive with targets (if train) and metadata.
    """
    # 1. Load Raw Data (GNSS + GT Targets aligned by time)
    # This uses the data_loader module which handles cleaning and target computation (ENU residuals)
    # For test set, gt_path is None, so it returns all GNSS epochs.
    raw_df = load_drive_data(
        drive_id=drive_id,
        phone_name=phone_name,
        gnss_rel_path=gnss_path,
        gt_rel_path=gt_path,
        load_cached_data=True,
    )

    if raw_df is None or raw_df.empty:
        return None

    # 2. Generate Kinematic Features using Dual-Projection Physics
    # This uses the gnss_physics module to compute forces (Pseudorange and Doppler projected residuals)
    features_df = generate_kinematic_features(
        gnss_df=raw_df, drive_id=drive_id, phone_name=phone_name, load_cached_data=True
    )

    if features_df is None or features_df.empty:
        return None

    # 3. Aggregate Raw Data to Epoch Level
    # raw_df has one row per satellite. We need one row per epoch to merge with features.
    # Targets (target_E, target_N, target_U) are constant for all satellites in an epoch.

    # Define columns to aggregate (taking the first value since they are constant per epoch)
    agg_dict = {}

    # Targets and GT info
    if is_train:
        targets = [
            "target_E",
            "target_N",
            "target_U",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
        ]
        for t in targets:
            if t in raw_df.columns:
                agg_dict[t] = "first"

    # WLS info (Reference position for the ENU projection)
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    for c in wls_cols:
        if c in raw_df.columns:
            agg_dict[c] = "first"

    # Determine the time column used for grouping
    # load_drive_data merges on 'UnixTimeMillis' (GT) or 'utcTimeMillis' (GNSS)
    time_col = (
        "UnixTimeMillis" if "UnixTimeMillis" in raw_df.columns else "utcTimeMillis"
    )

    if not agg_dict:
        # If no targets/WLS to aggregate (e.g. raw test data without WLS), just use time index
        epoch_df = pd.DataFrame(index=raw_df[time_col].unique())
        epoch_df.index.name = time_col
    else:
        epoch_df = raw_df.groupby(time_col).agg(agg_dict)

    # 4. Merge Features with Epoch Data
    # features_df index is utcTimeMillis. epoch_df index is the time column.
    # We assume they match (which they should as they come from the same source)
    merged_df = pd.merge(
        epoch_df, features_df, left_index=True, right_index=True, how="inner"
    )

    # Add Metadata
    merged_df["drive_id"] = drive_id
    merged_df["phone_name"] = phone_name
    merged_df["tripId"] = f"{drive_id}-{phone_name}"

    # Ensure the index is a column named UnixTimeMillis
    merged_df.index.name = "UnixTimeMillis"

    return merged_df.reset_index()


def create_dataset(split, load_cached_data=True):
    """
    Generates the complete dataset for a given split (train/val/test).

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load the full dataset from cache if available.

    Returns:
        pd.DataFrame: The complete dataset ready for model training/inference.
    """
    cache_path = os.path.join(CACHE_DIR, f"dataset_{split}.parquet")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} dataset from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating {split} dataset...")

    # Load Metadata
    meta_df = load_metadata(split)

    # Get unique drives to process
    # We group by drive and phone to avoid redundant processing of the same log file
    unique_trips = meta_df[
        ["drive_id", "phone_name", "gnss_path", "gt_path"]
    ].drop_duplicates()

    # Handle missing gt_path for test set
    if "gt_path" not in unique_trips.columns:
        unique_trips["gt_path"] = None

    drive_datasets = []
    for _, row in unique_trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]
        gt_path = (
            row["gt_path"] if "gt_path" in row and pd.notna(row["gt_path"]) else None
        )

        # Explicitly disable GT loading for test split
        if split == "test":
            gt_path = None

        df = process_drive(
            drive_id=drive_id,
            phone_name=phone_name,
            gnss_path=gnss_path,
            gt_path=gt_path,
            is_train=(split != "test"),
        )

        if df is not None:
            drive_datasets.append(df)

    if not drive_datasets:
        raise ValueError(f"No data generated for split: {split}")

    full_df = pd.concat(drive_datasets, ignore_index=True)

    # For Test set, filter/align to only the rows required by the sample submission
    if split == "test":
        # The submission file defines specific timestamps for specific trips
        req_indices = meta_df[["tripId", "UnixTimeMillis"]]

        # Merge to filter and align order.
        # Left join ensures we keep the submission structure.
        # Missing rows (if any) will have NaNs, which should be handled during inference.
        full_df = pd.merge(
            req_indices, full_df, on=["tripId", "UnixTimeMillis"], how="left"
        )

    # Save to cache
    print(f"Saving {split} dataset to {cache_path}")
    full_df.to_parquet(cache_path)

    return full_df
