import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.features import process_drive as process_features
from library.kinematics import CarrierPhaseOdometry


def load_dataset(split="train", max_drives=None, load_cached_data=True):
    """
    Loads and processes the dataset for a specific split (train, val, test).
    Combines ML features (Anchors) and Kinematics (Odometry) into a single DataFrame.

    Args:
        split (str): One of 'train', 'val', 'test'.
        max_drives (int, optional): Limit the number of drives to process (for debugging).
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    # 1. Define Cache Path
    cache_path = os.path.join(OUTPUT_DIR, f"dataset_{split}.parquet")

    # 2. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} dataset from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 3. Load Metadata
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Get unique drive-phone pairs
    # We use a list of tuples to iterate
    unique_trips = meta_df[["drive_id", "phone_name"]].drop_duplicates().values.tolist()

    if max_drives:
        unique_trips = unique_trips[:max_drives]

    print(f"Processing {len(unique_trips)} drives for {split} set...")

    # Initialize Kinematics Processor
    cpo = CarrierPhaseOdometry()

    processed_dfs = []

    for drive_id, phone_name in unique_trips:
        # Get paths from metadata (take first row for this trip)
        trip_meta = meta_df[
            (meta_df["drive_id"] == drive_id) & (meta_df["phone_name"] == phone_name)
        ].iloc[0]

        gnss_rel_path = trip_meta["gnss_path"]
        gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)

        gt_path = None
        if split in ["train", "val"]:
            gt_rel_path = trip_meta["gt_path"]
            gt_path = os.path.join(INPUT_DIR, gt_rel_path)

        # A. Process ML Features (Anchors)
        # This returns features + targets (res_E, res_N) + WLS positions
        df_features = process_features(
            drive_id=drive_id,
            phone_name=phone_name,
            gnss_path=gnss_path,
            gt_path=gt_path,
            load_cached_data=load_cached_data,
        )

        if df_features.empty:
            print(f"Warning: No features generated for {drive_id}-{phone_name}")
            continue

        # B. Process Kinematics (Odometry)
        # This returns [UnixTimeMillis, d_E, d_N, d_U, weight]
        df_kinematics = cpo.process_drive(
            drive_id=drive_id,
            phone_name=phone_name,
            gnss_path=gnss_path,
            load_cached_data=load_cached_data,
        )

        # C. Merge Streams
        # We merge kinematics onto features based on timestamp.
        # Features are point-wise (at time t). Kinematics are usually t-1 -> t.
        # Both dataframes are indexed by UnixTimeMillis.

        # Rename kinematics columns to avoid collision if necessary (though names are unique)
        # df_kinematics columns: UnixTimeMillis, d_E, d_N, d_U, weight

        if not df_kinematics.empty:
            df_merged = pd.merge(
                df_features, df_kinematics, on="UnixTimeMillis", how="left"
            )

            # Fill missing kinematics with 0 (start of track or gaps)
            # A weight of 0 indicates the graph optimizer should ignore the kinematic edge
            df_merged["d_E"] = df_merged["d_E"].fillna(0.0)
            df_merged["d_N"] = df_merged["d_N"].fillna(0.0)
            df_merged["d_U"] = df_merged["d_U"].fillna(0.0)
            df_merged["weight"] = df_merged["weight"].fillna(0.0)
        else:
            # If kinematics failed entirely, fill with zeros
            df_merged = df_features.copy()
            df_merged["d_E"] = 0.0
            df_merged["d_N"] = 0.0
            df_merged["d_U"] = 0.0
            df_merged["weight"] = 0.0

        # Add Metadata columns for grouping
        df_merged["drive_id"] = drive_id
        df_merged["phone_name"] = phone_name

        processed_dfs.append(df_merged)

    if not processed_dfs:
        print("No data processed.")
        return pd.DataFrame()

    # Concatenate all drives
    full_dataset = pd.concat(processed_dfs, ignore_index=True)

    # 4. Save to Cache
    print(f"Saving {split} dataset to {cache_path}...")
    full_dataset.to_parquet(cache_path)

    return full_dataset
