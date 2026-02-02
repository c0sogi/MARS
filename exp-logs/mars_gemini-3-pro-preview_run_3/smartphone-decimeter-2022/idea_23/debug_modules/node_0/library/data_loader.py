import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)


class GnssLoader:
    """
    Handles ingestion of raw GNSS, IMU, and Ground Truth data with caching.
    """

    def __init__(self, input_dir=INPUT_DIR, working_dir=WORKING_DIR):
        self.input_dir = input_dir
        self.working_dir = working_dir
        # Create a subdirectory for raw data cache to keep working dir organized
        self.cache_dir = os.path.join(working_dir, "raw_data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, rel_path):
        """
        Generates a safe cache filename from the relative file path.
        Example: train/2020.../device_gnss.csv -> train_2020..._device_gnss.parquet
        """
        safe_name = rel_path.replace(os.sep, "_").replace(".csv", ".parquet")
        return os.path.join(self.cache_dir, safe_name)

    def _load_csv_with_cache(self, rel_path, load_cached_data=True):
        """
        Generic loader that checks for a cached parquet file before reading CSV.
        """
        full_csv_path = os.path.join(self.input_dir, rel_path)
        cache_path = self._get_cache_path(rel_path)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Using pandas read_parquet (requires pyarrow or fastparquet)
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load cache {cache_path}: {e}. Falling back to CSV."
                )

        # 2. Load from CSV if cache missing or failed
        if not os.path.exists(full_csv_path):
            raise FileNotFoundError(f"Source file not found: {full_csv_path}")

        # low_memory=False prevents mixed type inference warnings for large files
        df = pd.read_csv(full_csv_path, low_memory=False)

        # 3. Save to cache for next time
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to write cache {cache_path}: {e}")

        return df

    def load_metadata(self, split="train"):
        """
        Load the metadata CSV for the specified split.
        """
        if split == "train":
            path = TRAIN_METADATA_PATH
        elif split == "val":
            path = VAL_METADATA_PATH
        elif split == "test":
            path = TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Metadata file not found at {path}. Please run metadata generation first."
            )

        return pd.read_csv(path)

    def load_gnss(self, rel_path, load_cached_data=True):
        """
        Load GNSS log data.
        """
        return self._load_csv_with_cache(rel_path, load_cached_data=load_cached_data)

    def load_imu(self, rel_path, load_cached_data=True):
        """
        Load IMU sensor data.
        """
        return self._load_csv_with_cache(rel_path, load_cached_data=load_cached_data)

    def load_ground_truth(self, rel_path, load_cached_data=True):
        """
        Load Ground Truth data.
        """
        return self._load_csv_with_cache(rel_path, load_cached_data=load_cached_data)

    def clean_gnss(self, gnss_df):
        """
        Apply basic cleaning to GNSS data.
        - Ensures critical columns are numeric.
        - Drops rows with missing timing information.
        """
        # Critical columns for processing
        numeric_cols = [
            "TimeNanos",
            "FullBiasNanos",
            "BiasNanos",
            "Cn0DbHz",
            "RawPseudorangeMeters",
            "utcTimeMillis",
        ]

        # Enforce numeric types
        for col in numeric_cols:
            if col in gnss_df.columns:
                gnss_df[col] = pd.to_numeric(gnss_df[col], errors="coerce")

        # Drop rows where TimeNanos is NaN (cannot align or process)
        if "TimeNanos" in gnss_df.columns:
            gnss_df = gnss_df.dropna(subset=["TimeNanos"])

        return gnss_df

    def get_drive_data(
        self, drive_id, phone_name, split="train", load_cached_data=True
    ):
        """
        Retrieves all dataframes (GNSS, IMU, GT) for a specific drive and phone.
        Uses metadata to resolve file paths.

        Returns:
            gnss_df (pd.DataFrame): Cleaned GNSS data.
            imu_df (pd.DataFrame): Raw IMU data.
            gt_df (pd.DataFrame or None): Ground truth data (None if test split).
        """
        # Load metadata to find paths
        meta_df = self.load_metadata(split)

        # Filter for the specific drive/phone
        # We only need one row to get the file paths as they are constant for the trip
        subset = meta_df[
            (meta_df["drive_id"] == drive_id) & (meta_df["phone_name"] == phone_name)
        ]

        if subset.empty:
            raise ValueError(
                f"No metadata entry found for Drive: {drive_id}, Phone: {phone_name} in {split} split."
            )

        # Extract paths from the first record
        row = subset.iloc[0]
        gnss_path = row["gnss_path"]
        imu_path = row["imu_path"]

        # Load Data
        gnss_df = self.load_gnss(gnss_path, load_cached_data=load_cached_data)
        imu_df = self.load_imu(imu_path, load_cached_data=load_cached_data)

        # Clean GNSS
        gnss_df = self.clean_gnss(gnss_df)

        # Load Ground Truth if available (Train/Val)
        gt_df = None
        if "gt_path" in row and pd.notna(row["gt_path"]):
            gt_df = self.load_ground_truth(
                row["gt_path"], load_cached_data=load_cached_data
            )

        return gnss_df, imu_df, gt_df
