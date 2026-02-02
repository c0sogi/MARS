import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    setup_logger,
    reduce_mem_usage,
    save_to_parquet,
    load_from_parquet,
)


class DataLoader:
    """
    Handles loading, preprocessing, and merging of metadata and tracking data.
    Implements caching to speed up iterative development.
    """

    def __init__(self, debug: bool = Config.DEBUG):
        """
        Args:
            debug (bool): If True, loads a smaller subset of data for debugging.
        """
        self.debug = debug
        self.logger = setup_logger(name="DataLoader")

    def load_metadata(self, split: str) -> pd.DataFrame:
        """
        Loads the metadata CSV for the specified split.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: The loaded metadata.
        """
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        self.logger.info(f"Loading {split} metadata from {path}...")
        df = pd.read_csv(path)

        if self.debug and len(df) > Config.DEBUG_SAMPLE_SIZE:
            self.logger.info(
                f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from metadata."
            )
            df = df.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

        # Ensure join keys are consistent
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)
        # nfl_player_id_2 remains object because it can be 'G'

        return reduce_mem_usage(df)

    def load_tracking(self, split: str) -> pd.DataFrame:
        """
        Loads the tracking data CSV corresponding to the split.
        Note: Train and Val splits share the 'train' tracking file.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: The loaded tracking data.
        """
        if split in ["train", "val"]:
            path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        self.logger.info(f"Loading tracking data from {path}...")
        # Only load necessary columns to save memory
        cols_to_load = ["game_play", "step", "nfl_player_id"] + Config.TRACKING_COLS
        df = pd.read_csv(path, usecols=cols_to_load)

        # Ensure join keys are consistent
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)
        df["nfl_player_id"] = df["nfl_player_id"].astype(int)

        return reduce_mem_usage(df)

    def merge_tracking_data(
        self, metadata_df: pd.DataFrame, tracking_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges tracking data onto metadata for both Player 1 and Player 2.
        Handles 'G' (Ground) for Player 2 by leaving tracking features as NaN.

        Args:
            metadata_df (pd.DataFrame): The base metadata.
            tracking_df (pd.DataFrame): The player tracking data.

        Returns:
            pd.DataFrame: Merged DataFrame with _p1 and _p2 feature columns.
        """
        self.logger.info("Merging tracking data...")

        # Filter tracking data to only include game_plays present in metadata
        # This optimization reduces the size of the right table in the join
        relevant_plays = metadata_df["game_play"].unique()
        tracking_subset = tracking_df[
            tracking_df["game_play"].isin(relevant_plays)
        ].copy()

        # --- Merge Player 1 ---
        self.logger.info("Merging Player 1 tracking data...")
        merged_df = pd.merge(
            metadata_df,
            tracking_subset,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P1 columns
        rename_dict_p1 = {col: f"{col}_p1" for col in Config.TRACKING_COLS}
        merged_df.rename(columns=rename_dict_p1, inplace=True)
        merged_df.drop(
            columns=["nfl_player_id"], inplace=True
        )  # Drop the join key from right table

        # --- Merge Player 2 ---
        self.logger.info("Merging Player 2 tracking data...")

        # Create a temporary numeric column for join, coercing 'G' to NaN
        merged_df["nfl_player_id_2_int"] = pd.to_numeric(
            merged_df["nfl_player_id_2"], errors="coerce"
        )

        merged_df = pd.merge(
            merged_df,
            tracking_subset,
            left_on=["game_play", "step", "nfl_player_id_2_int"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P2 columns
        rename_dict_p2 = {col: f"{col}_p2" for col in Config.TRACKING_COLS}
        merged_df.rename(columns=rename_dict_p2, inplace=True)

        # Cleanup
        merged_df.drop(columns=["nfl_player_id", "nfl_player_id_2_int"], inplace=True)

        return reduce_mem_usage(merged_df)

    def get_merged_data(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Orchestrates loading and merging with caching support.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The final merged dataframe.
        """
        # Define cache filename based on split and debug status
        debug_suffix = "_debug" if self.debug else ""
        cache_filename = f"merged_{split}{debug_suffix}.parquet"

        # 1. Try to load from cache
        if load_cached_data:
            cached_df = load_from_parquet(cache_filename)
            if cached_df is not None:
                self.logger.info(f"Loaded {split} data from cache: {cache_filename}")
                return cached_df
            else:
                self.logger.info(
                    f"Cache not found for {cache_filename}. Processing from scratch..."
                )
        else:
            self.logger.info(f"Ignoring cache. Processing {split} data from scratch...")

        # 2. Process from scratch
        metadata_df = self.load_metadata(split)
        tracking_df = self.load_tracking(split)

        merged_df = self.merge_tracking_data(metadata_df, tracking_df)

        # 3. Save to cache
        self.logger.info(f"Saving merged {split} data to cache: {cache_filename}")
        save_to_parquet(merged_df, cache_filename)

        return merged_df
