import os
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    """
    Handles the ingestion of raw data files for the Doppler-Aided Residual Boosting pipeline.
    Implements caching to Parquet to speed up subsequent data loading.
    """

    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.metadata_dir = Config.METADATA_DIR
        self.working_dir = Config.WORKING_DIR

    def load_metadata(self, split: str):
        """
        Load metadata for a specific split.

        Args:
            split (str): 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: Metadata dataframe.
        """
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        return pd.read_csv(path)

    def _get_cache_path(self, split: str, sensor: str) -> str:
        """
        Construct the path for the cached parquet file.
        """
        filename = f"{split}_{sensor}_raw.parquet"
        return os.path.join(self.working_dir, filename)

    def load_sensor_data(
        self, split: str, sensor: str, load_cached_data: bool = True, limit: int = None
    ):
        """
        Load and aggregate raw sensor data (GNSS, IMU, or GT) for a given split.
        Implements caching using Parquet.

        Args:
            split (str): 'train', 'val', or 'test'.
            sensor (str): 'gnss', 'imu', or 'gt' (ground truth).
            load_cached_data (bool): If True, attempt to load from cache first.
            limit (int, optional): Limit the number of unique trips processed (for debugging).

        Returns:
            pd.DataFrame: Aggregated sensor data.
        """
        cache_path = self._get_cache_path(split, sensor)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached {sensor.upper()} data for {split} from {cache_path}..."
            )
            try:
                df = pd.read_parquet(cache_path)
                # If a limit was applied during caching, the file might be smaller than full.
                # However, usually we cache the full dataset.
                # If limit is requested NOW, we slice the loaded df.
                # But to be safe and deterministic, if limit is provided, we might want to reload
                # or just slice based on unique trips in the cached df.
                # For simplicity, if cache exists, we assume it's what the user wants,
                # unless they explicitly set load_cached_data=False.
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing raw {sensor.upper()} data for {split} (Limit: {limit})...")

        # Load metadata to get file paths
        meta_df = self.load_metadata(split)

        # Determine the column containing the path
        if sensor == "gnss":
            path_col = "gnss_path"
        elif sensor == "imu":
            path_col = "imu_path"
        elif sensor == "gt":
            path_col = "gt_path"
            if split == "test":
                print(
                    "Warning: Ground Truth requested for test set. Returning empty DataFrame."
                )
                return pd.DataFrame()
        else:
            raise ValueError(f"Invalid sensor type: {sensor}")

        # Get unique trips/paths
        # We process by unique file path to avoid reading the same file multiple times
        # (Metadata has one row per timestamp, so many rows share the same file path)
        unique_paths_df = meta_df[
            ["drive_id", "phone_name", path_col]
        ].drop_duplicates()

        if limit:
            unique_paths_df = unique_paths_df.head(limit)
            print(f"Limiting to {limit} unique files.")

        data_frames = []

        for _, row in unique_paths_df.iterrows():
            rel_path = row[path_col]
            drive_id = row["drive_id"]
            phone_name = row["phone_name"]

            full_path = os.path.join(self.input_dir, rel_path)

            if os.path.exists(full_path):
                try:
                    # Read CSV
                    df_chunk = pd.read_csv(full_path)

                    # Add identifier columns to join later if needed
                    df_chunk["drive_id"] = drive_id
                    df_chunk["phone_name"] = phone_name

                    # For GNSS/IMU, we might want to ensure 'tripId' exists or can be constructed.
                    # The metadata constructs tripId as f"{drive_id}-{phone_name}".
                    df_chunk["tripId"] = f"{drive_id}-{phone_name}"

                    data_frames.append(df_chunk)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
            else:
                print(f"Warning: File not found: {full_path}")

        if not data_frames:
            print(f"No data found for {split} {sensor}.")
            return pd.DataFrame()

        # Concatenate all chunks
        aggregated_df = pd.concat(data_frames, ignore_index=True)

        # 3. Save to cache
        # Ensure working directory exists (handled in Config, but good to be safe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        print(f"Saving aggregated {sensor.upper()} data to {cache_path}...")
        aggregated_df.to_parquet(cache_path, index=False)

        return aggregated_df

    def load_ground_truth(
        self, split: str, load_cached_data: bool = True, limit: int = None
    ):
        """
        Wrapper to load Ground Truth data.
        """
        return self.load_sensor_data(split, "gt", load_cached_data, limit)

    def load_gnss(self, split: str, load_cached_data: bool = True, limit: int = None):
        """
        Wrapper to load GNSS data.
        """
        return self.load_sensor_data(split, "gnss", load_cached_data, limit)

    def load_imu(self, split: str, load_cached_data: bool = True, limit: int = None):
        """
        Wrapper to load IMU data.
        """
        return self.load_sensor_data(split, "imu", load_cached_data, limit)
