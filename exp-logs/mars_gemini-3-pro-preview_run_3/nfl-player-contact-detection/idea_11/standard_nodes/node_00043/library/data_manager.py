import pandas as pd
import numpy as np
import os
import gc
from library.config import Config


class DataManager:
    """
    Manages the loading, filtering, and caching of raw datasets for the Contact Detection pipeline.
    Handles Metadata, Player Tracking, and Helmet Baseline data.
    """

    def __init__(self):
        """
        Initialize the DataManager. Ensures working directories exist.
        """
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def load_metadata(self, split: str) -> pd.DataFrame:
        """
        Loads the metadata (labels) for a specific split.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe with standardized columns.
        """
        if split == "train":
            path = Config.TRAIN_META_PATH
        elif split == "val":
            path = Config.VAL_META_PATH
        elif split == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Standardize Join Keys
        df["game_play"] = df["game_play"].astype(str)

        # Player 1 is always a player ID (int)
        df["nfl_player_id_1"] = (
            pd.to_numeric(df["nfl_player_id_1"], errors="coerce").fillna(-1).astype(int)
        )

        # Player 2 can be a player ID or 'G' (Ground). Keep as string to preserve 'G'.
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

        return df

    def load_tracking(
        self, split: str, game_plays: list, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads and filters player tracking data.

        Args:
            split (str): 'train', 'val', or 'test'.
            game_plays (list): List of game_play IDs to keep.
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: Filtered tracking data.
        """
        cache_path = os.path.join(self.working_dir, f"raw_tracking_{split}.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        # 2. Determine source file
        # Note: 'val' split comes from the training source file
        if split in ["train", "val"]:
            source_path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            source_path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # 3. Load and Filter
        # Reading full CSV is acceptable given 220GB RAM, but we filter immediately.
        # Specifying dtypes helps memory slightly.
        dtype_spec = {
            "game_play": "str",
            "nfl_player_id": "float32",  # float to handle potential NaNs before cast
            "step": "int16",
            "position": "category",
            "team": "category",
        }

        df = pd.read_csv(source_path, dtype=dtype_spec)

        # Filter by game_play
        df = df[df["game_play"].isin(game_plays)].copy()

        # Standardize IDs
        # Fill NaNs with -1 and cast to int
        df["nfl_player_id"] = df["nfl_player_id"].fillna(-1).astype(int)

        # 4. Save to cache
        df.to_parquet(cache_path, index=False)

        return df

    def load_helmets(
        self, split: str, game_plays: list, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads and filters baseline helmet detection data.

        Args:
            split (str): 'train', 'val', or 'test'.
            game_plays (list): List of game_play IDs to keep.
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: Filtered helmet data.
        """
        cache_path = os.path.join(self.working_dir, f"raw_helmets_{split}.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        # 2. Determine source file
        if split in ["train", "val"]:
            source_path = Config.TRAIN_HELMETS_PATH
        elif split == "test":
            source_path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # 3. Load and Filter
        # Helmet file is large (~3.4M rows), but manageable.
        df = pd.read_csv(source_path)

        # Filter by game_play
        df = df[df["game_play"].isin(game_plays)].copy()

        # Standardize IDs
        # Helmet predictions might have missing player IDs or NaNs
        df["nfl_player_id"] = (
            pd.to_numeric(df["nfl_player_id"], errors="coerce").fillna(-1).astype(int)
        )

        # Ensure view is categorical for memory efficiency
        df["view"] = df["view"].astype("category")

        # 4. Save to cache
        df.to_parquet(cache_path, index=False)

        return df

    def get_data(self, split: str, load_cached_data: bool = True):
        """
        Orchestrates the loading of all necessary raw data for a given split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached intermediate files.

        Returns:
            tuple: (metadata_df, tracking_df, helmets_df)
        """
        # 1. Load Metadata (Labels/Split definition)
        # Metadata is small and fast, no need for extra caching layer here
        meta_df = self.load_metadata(split)

        # Extract relevant game_plays to filter heavy datasets
        unique_game_plays = meta_df["game_play"].unique().tolist()

        # 2. Load Tracking Data
        tracking_df = self.load_tracking(split, unique_game_plays, load_cached_data)

        # 3. Load Helmet Data
        helmets_df = self.load_helmets(split, unique_game_plays, load_cached_data)

        return meta_df, tracking_df, helmets_df
