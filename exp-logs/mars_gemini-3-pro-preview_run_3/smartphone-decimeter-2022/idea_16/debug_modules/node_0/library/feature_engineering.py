import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_loader import (
    load_drive_data,
    process_drive,
    compute_geometry_features,
)


class GeometryFeatureExtractor:
    """
    Wraps the physics-based geometry feature extraction logic.
    Calculates Line-of-Sight vectors, Geometry Matrix, and Net Error Force.
    """

    def __init__(self):
        pass

    def transform(self, df_gnss):
        """
        Compute geometry features from raw GNSS data.

        Args:
            df_gnss (pd.DataFrame): Raw GNSS measurements.

        Returns:
            pd.DataFrame: Aggregated features including Force vectors and Geometry matrix diagonals.
        """
        return compute_geometry_features(df_gnss)


def get_processed_dataset(
    split: str, load_cached_data: bool = True, max_drives: int = None
) -> pd.DataFrame:
    """
    Generates or loads the processed dataset for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, tries to load from parquet cache.
        max_drives (int, optional): Limit the number of drives processed (for debugging).
                                    If set, caching is skipped to avoid overwriting full data with partial data.

    Returns:
        pd.DataFrame: Processed dataset with features and targets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_dataset.parquet")

    # 1. Try Cache (only if not debugging with max_drives)
    if load_cached_data and max_drives is None and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating {split} data from raw files (max_drives={max_drives})...")

    # 2. Load Metadata to identify drives
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Get unique drive-phone pairs
    groups = meta_df[["drive_id", "phone_name"]].drop_duplicates()

    # Apply max_drives limit for debugging
    if max_drives is not None:
        groups = groups.head(max_drives)

    all_data = []

    for _, row in groups.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Load Raw Data
        df_gnss, df_imu, df_gt = load_drive_data(drive_id, phone_name)

        if df_gnss is None or df_gnss.empty:
            continue

        # Process Drive (Feature Engineering + Target Alignment)
        # process_drive handles GNSS/IMU feature extraction, ENU rotation, and GT merging
        processed_df = process_drive(drive_id, phone_name, df_gnss, df_imu, df_gt)

        if processed_df is not None and not processed_df.empty:
            # Filter to only rows present in metadata (requested timestamps)
            target_timestamps = meta_df[
                (meta_df["drive_id"] == drive_id)
                & (meta_df["phone_name"] == phone_name)
            ]["UnixTimeMillis"].values

            processed_df = processed_df[
                processed_df["UnixTimeMillis"].isin(target_timestamps)
            ].copy()

            # Add identifiers
            processed_df["drive_id"] = drive_id
            processed_df["phone_name"] = phone_name

            all_data.append(processed_df)

    if not all_data:
        if max_drives is not None:
            print(
                f"Warning: No data generated for split {split} with max_drives={max_drives}"
            )
            return pd.DataFrame()
        raise ValueError(f"No data generated for split {split}")

    final_df = pd.concat(all_data, ignore_index=True)

    # 3. Save Cache (only if full dataset is generated)
    if max_drives is None:
        print(f"Saving {split} data to cache: {cache_path}")
        final_df.to_parquet(cache_path, index=False)

    return final_df
