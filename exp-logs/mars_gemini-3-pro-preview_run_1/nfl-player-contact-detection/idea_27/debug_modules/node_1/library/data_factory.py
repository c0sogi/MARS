import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import CacheManager, seed_everything


class DataFactory:
    """
    Handles loading of raw data and merging of tracking information.
    Implements caching to optimize runtime for iterative development.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        seed_everything(Config.SEED)

    def load_metadata(self, split="train"):
        """
        Loads the metadata CSV for the specified split.

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
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        return pd.read_csv(path)

    def load_tracking(self, split="train"):
        """
        Loads the player tracking data.

        Args:
            split (str): 'train' (for train/val) or 'test'.

        Returns:
            pd.DataFrame: The tracking dataframe containing only necessary columns.
        """
        if split in ["train", "val"]:
            path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found at {path}")

        # Read only the columns needed for merging and feature engineering
        # Keys: game_play, step, nfl_player_id
        # Features: Config.TRACKING_COLS
        cols_to_read = list(
            set(["game_play", "step", "nfl_player_id"] + Config.TRACKING_COLS)
        )

        return pd.read_csv(path, usecols=cols_to_read)

    def merge_tracking_data(
        self, metadata_df, tracking_df, split="train", load_cached_data=True
    ):
        """
        Merges metadata with tracking data for Player 1 and Player 2.
        Handles the 'G' (Ground) case for Player 2.
        Caches the result to disk.

        Args:
            metadata_df (pd.DataFrame): Metadata containing contact pairs.
            tracking_df (pd.DataFrame): Player tracking data.
            split (str): Split name used for cache key generation.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: Merged dataframe with suffixes _p1 and _p2.
        """
        cache_filename = f"merged_{split}.parquet"

        # 1. Check Cache
        if load_cached_data and self.cache_manager.exists(cache_filename):
            # print(f"Loading merged {split} data from cache...")
            return self.cache_manager.load_parquet(cache_filename)

        # print(f"Merging tracking data for {split}...")

        # 2. Type Enforcement for Merge Keys
        # Metadata
        metadata_df["game_play"] = metadata_df["game_play"].astype(str)
        metadata_df["step"] = metadata_df["step"].astype(int)
        metadata_df["nfl_player_id_1"] = metadata_df["nfl_player_id_1"].astype(int)

        # Tracking
        tracking_df["game_play"] = tracking_df["game_play"].astype(str)
        tracking_df["step"] = tracking_df["step"].astype(int)
        tracking_df["nfl_player_id"] = tracking_df["nfl_player_id"].astype(int)

        # 3. Handle Player 2 ID (Mixed Types: Int and 'G')
        # Create a numeric column for merging, forcing 'G' to NaN
        metadata_df["p2_merge_id"] = pd.to_numeric(
            metadata_df["nfl_player_id_2"], errors="coerce"
        )

        # Identify feature columns to rename (exclude keys)
        feature_cols = [
            c
            for c in tracking_df.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]

        # 4. Merge Player 1
        merged_df = pd.merge(
            metadata_df,
            tracking_df,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P1 columns
        rename_map_p1 = {col: f"{col}_p1" for col in feature_cols}
        merged_df = merged_df.rename(columns=rename_map_p1)
        merged_df = merged_df.drop(columns=["nfl_player_id"], errors="ignore")

        # 5. Merge Player 2
        # Note: Rows where p2_merge_id is NaN (Ground) will result in NaN for _p2 columns
        merged_df = pd.merge(
            merged_df,
            tracking_df,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P2 columns
        rename_map_p2 = {col: f"{col}_p2" for col in feature_cols}
        merged_df = merged_df.rename(columns=rename_map_p2)

        # Cleanup
        merged_df = merged_df.drop(
            columns=["nfl_player_id", "p2_merge_id"], errors="ignore"
        )

        # 6. Save to Cache
        self.cache_manager.save_parquet(merged_df, cache_filename)

        return merged_df

    def get_data(self, split="train", load_cached_data=True):
        """
        Convenience method to load and merge data for a specific split.
        """
        meta = self.load_metadata(split)
        track = self.load_tracking(split)
        return self.merge_tracking_data(meta, track, split, load_cached_data)
