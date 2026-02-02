import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    CACHE_DIR,
)
from library.utils import setup_logger, CacheManager


class DataProcessor:
    """
    Handles data ingestion, preprocessing, and merging of metadata with tracking data.
    """

    def __init__(self, logger=None):
        self.logger = (
            logger
            if logger
            else setup_logger(os.path.join(os.getcwd(), "logs", "data_processing.log"))
        )
        self.cache_manager = CacheManager()

    def load_metadata(self, split="train", sample_size=None):
        """
        Loads the metadata CSV for the specified split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            sample_size (int, optional): If provided, samples the dataset for debugging.

        Returns:
            pd.DataFrame: The loaded metadata.
        """
        self.logger.info(f"Loading metadata for split: {split}")

        if split == "train":
            path = TRAIN_METADATA_PATH
        elif split == "val":
            path = VAL_METADATA_PATH
        elif split == "test":
            path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        df = pd.read_csv(path)

        # Ensure step is int
        if "step" in df.columns:
            df["step"] = df["step"].astype(int)

        if sample_size is not None and sample_size < len(df):
            self.logger.info(f"Sampling {sample_size} rows from {split} metadata.")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        self.logger.info(f"Loaded {len(df)} rows for {split} metadata.")
        return df

    def load_tracking(self, split="train"):
        """
        Loads the tracking data CSV.

        Args:
            split (str): One of 'train', 'val', 'test'.
                       Note: 'val' uses the 'train' tracking file.

        Returns:
            pd.DataFrame: The loaded tracking data.
        """
        self.logger.info(f"Loading tracking data for split: {split}")

        if split in ["train", "val"]:
            path = TRAIN_TRACKING_PATH
        elif split == "test":
            path = TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found at {path}")

        # Load tracking data
        # We specify dtypes to optimize memory and ensure consistency
        df = pd.read_csv(path)

        # Ensure key columns are correct types
        df["step"] = df["step"].astype(int)
        df["nfl_player_id"] = df["nfl_player_id"].astype(int)
        df["game_play"] = df["game_play"].astype(str)

        self.logger.info(f"Loaded {len(df)} rows of tracking data.")
        return df

    def merge_tracking_data(
        self, df_meta, df_tracking, split="train", load_cached_data=True
    ):
        """
        Merges metadata with tracking data for both Player 1 and Player 2.
        Implements caching to avoid re-computation.

        Args:
            df_meta (pd.DataFrame): Metadata dataframe.
            df_tracking (pd.DataFrame): Tracking dataframe.
            split (str): Split name used for cache filename.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Merged dataframe with tracking features for p1 and p2.
        """
        cache_filename = f"merged_{split}.parquet"

        # 1. Try to load from cache
        if load_cached_data:
            cached_df = self.cache_manager.load_parquet(cache_filename)
            if cached_df is not None:
                self.logger.info(f"Loaded merged data from cache: {cache_filename}")
                return cached_df

        self.logger.info(
            f"Merging tracking data for {split} (Cache miss or force reload)..."
        )

        # Prepare Metadata
        # Ensure join keys match tracking data types
        df_meta = df_meta.copy()
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

        # Handle Player 2 (which can be 'G')
        # Create a numeric column for merging, coercing 'G' to NaN
        df_meta["nfl_player_id_2_numeric"] = pd.to_numeric(
            df_meta["nfl_player_id_2"], errors="coerce"
        )

        # Prepare Tracking Data (subset columns if needed, but we take all for now)
        # We drop 'datetime' from tracking to avoid collision/confusion with metadata 'datetime'
        track_cols = [c for c in df_tracking.columns if c != "datetime"]
        df_track_clean = df_tracking[track_cols].copy()

        # ---------------------------------------------------------
        # Merge Player 1
        # ---------------------------------------------------------
        self.logger.info("Merging Player 1 tracking data...")
        df_merged = pd.merge(
            df_meta,
            df_track_clean,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1_temp"),  # Handle collisions if any
        )

        # Rename columns for P1
        # The merge might result in columns like 'x_position' (from tracking).
        # We want 'x_position_p1'.
        # Identify columns that came from tracking (excluding keys)
        tracking_feature_cols = [
            c
            for c in df_track_clean.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]

        rename_dict_p1 = {col: f"{col}_p1" for col in tracking_feature_cols}
        df_merged.rename(columns=rename_dict_p1, inplace=True)

        # Drop redundant nfl_player_id from merge
        if "nfl_player_id" in df_merged.columns:
            df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # ---------------------------------------------------------
        # Merge Player 2
        # ---------------------------------------------------------
        self.logger.info("Merging Player 2 tracking data...")
        # We merge on the numeric version of player 2 ID.
        # Rows where p2='G' will have NaN in 'nfl_player_id_2_numeric', so merge will result in NaNs for p2 features.

        df_merged = pd.merge(
            df_merged,
            df_track_clean,
            left_on=["game_play", "step", "nfl_player_id_2_numeric"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2_temp"),
        )

        # Rename columns for P2
        rename_dict_p2 = {col: f"{col}_p2" for col in tracking_feature_cols}
        df_merged.rename(columns=rename_dict_p2, inplace=True)

        # Cleanup
        cols_to_drop = ["nfl_player_id", "nfl_player_id_2_numeric"]
        df_merged.drop(
            columns=[c for c in cols_to_drop if c in df_merged.columns], inplace=True
        )

        # ---------------------------------------------------------
        # Save to Cache
        # ---------------------------------------------------------
        self.logger.info(f"Saving merged data to cache: {cache_filename}")
        self.cache_manager.save_parquet(df_merged, cache_filename)

        return df_merged
