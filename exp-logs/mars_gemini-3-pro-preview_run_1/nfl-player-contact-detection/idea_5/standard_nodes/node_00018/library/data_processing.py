import os
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    def __init__(self):
        """
        Initializes the DataLoader with the configuration.
        Ensures the working directory exists.
        """
        self.config = Config
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def load_metadata(self, split="train", sample_size=None):
        """
        Loads the metadata for the specified split.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            sample_size (int, optional): Number of rows to sample for debugging.
                                         If None, loads the full dataset.

        Returns:
            pd.DataFrame: The loaded (and optionally sampled) metadata.
        """
        if split == "train":
            path = self.config.TRAIN_METADATA_PATH
        elif split == "val":
            path = self.config.VAL_METADATA_PATH
        elif split == "test":
            path = self.config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split provided: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        print(f"Loading {split} metadata from {path}...")
        df = pd.read_csv(path)

        if sample_size is not None and sample_size < len(df):
            print(f"Sampling {sample_size} rows from {split} metadata...")
            df = df.sample(n=sample_size, random_state=self.config.SEED).reset_index(
                drop=True
            )

        return df

    def load_tracking(self, split="train"):
        """
        Loads the player tracking data corresponding to the split.

        Args:
            split (str): 'train' (used for train/val) or 'test'.

        Returns:
            pd.DataFrame: The tracking data.
        """
        if split in ["train", "val"]:
            path = self.config.TRAIN_TRACKING_PATH
        elif split == "test":
            path = self.config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split provided: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found at {path}")

        print(f"Loading tracking data from {path}...")
        df = pd.read_csv(path)
        return df

    def merge_tracking_data(
        self, df_meta, df_tracking, split_name, load_cached_data=True
    ):
        """
        Merges metadata with tracking data for Player 1 and Player 2.
        Handles caching to disk using Parquet to save time on subsequent runs.

        Args:
            df_meta (pd.DataFrame): The metadata dataframe (labels).
            df_tracking (pd.DataFrame): The player tracking dataframe.
            split_name (str): The name of the split (e.g., 'train', 'val', 'test') used for cache naming.
            load_cached_data (bool): If True, attempts to load from cache before processing.

        Returns:
            pd.DataFrame: The merged dataframe with _p1 and _p2 tracking features.
        """
        # Construct a cache filename that is specific to the split and the data size
        # This prevents loading a full cache for a sampled run or vice versa.
        meta_len = len(df_meta)
        cache_filename = f"merged_{split_name}_{meta_len}.parquet"
        cache_path = os.path.join(self.config.WORKING_DIR, cache_filename)

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Cache hit! Loading merged data from {cache_path}")
            try:
                df_merged = pd.read_parquet(cache_path)
                return df_merged
            except Exception as e:
                print(f"Error loading cache ({e}). Recomputing...")

        print(f"Cache miss. Merging tracking data for {len(df_meta)} interactions...")

        # 2. Prepare Tracking Data
        # Filter strictly for necessary columns to reduce memory footprint
        # We assume standard column names from description
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]

        # Intersect with available columns to be safe
        available_cols = [c for c in track_cols if c in df_tracking.columns]
        df_track_sub = df_tracking[available_cols].copy()

        # 3. Prepare Metadata
        df_merged = df_meta.copy()

        # Handle Ground Interactions:
        # nfl_player_id_2 contains 'G'. We create a numeric join column where 'G' becomes NaN.
        # This ensures the merge for P2 yields NaNs for tracking features when it is Ground.
        df_merged["nfl_player_id_2_join"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        # Explicit flag for ground contact
        df_merged["is_ground"] = (df_merged["nfl_player_id_2"] == "G").astype(int)

        # 4. Merge Player 1
        # Left join on [game_play, step, nfl_player_id_1]
        df_merged = pd.merge(
            df_merged,
            df_track_sub,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Drop the duplicate nfl_player_id column from tracking (it matches nfl_player_id_1)
        if "nfl_player_id" in df_merged.columns:
            df_merged = df_merged.drop(columns=["nfl_player_id"])

        # Rename tracking features to _p1
        feature_cols = [
            c for c in available_cols if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_map_p1 = {c: f"{c}_p1" for c in feature_cols}
        df_merged = df_merged.rename(columns=rename_map_p1)

        # 5. Merge Player 2
        # Left join on [game_play, step, nfl_player_id_2_join]
        df_merged = pd.merge(
            df_merged,
            df_track_sub,
            left_on=["game_play", "step", "nfl_player_id_2_join"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Drop duplicate nfl_player_id
        if "nfl_player_id" in df_merged.columns:
            df_merged = df_merged.drop(columns=["nfl_player_id"])

        # Rename tracking features to _p2
        rename_map_p2 = {c: f"{c}_p2" for c in feature_cols}
        df_merged = df_merged.rename(columns=rename_map_p2)

        # Clean up temporary join column
        df_merged = df_merged.drop(columns=["nfl_player_id_2_join"])

        # 6. Save to Cache
        print(f"Saving merged data to {cache_path}...")
        df_merged.to_parquet(cache_path, index=False)

        return df_merged
