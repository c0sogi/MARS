import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage


class DataManager:
    """
    Handles data loading, merging, and caching for the NFL Contact Detection task.
    """

    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def load_dataset(self, mode="train", load_cached_data=True):
        """
        Loads the dataset for the specified mode.

        Args:
            mode (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The merged dataframe containing labels and tracking data.
        """
        if mode not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'validation', or 'test'."
            )

        cache_path = os.path.join(self.working_dir, f"merged_data_{mode}.parquet")

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{mode}] Loading cached data from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return reduce_mem_usage(df)

        print(f"[{mode}] Cache not found or ignored. Processing raw data...")

        # 2. Determine File Paths
        if mode == "train":
            meta_path = self.config.PATH_CONFIG["metadata_train"]
            tracking_path = self.config.PATH_CONFIG["train_tracking"]
        elif mode == "validation":
            meta_path = self.config.PATH_CONFIG["metadata_val"]
            # Validation subset comes from the training set, so we use train tracking
            tracking_path = self.config.PATH_CONFIG["train_tracking"]
        else:  # test
            meta_path = self.config.PATH_CONFIG["metadata_test"]
            tracking_path = self.config.PATH_CONFIG["test_tracking"]

        # 3. Load Metadata (Labels)
        print(f"[{mode}] Loading metadata from {meta_path}...")
        df_labels = pd.read_csv(meta_path)

        # Ensure ID columns are consistent strings for merging
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)

        # 4. Load Tracking Data
        print(f"[{mode}] Loading tracking data from {tracking_path}...")
        # Only load necessary columns to save memory
        tracking_cols = [
            "game_play",
            "step",
            "nfl_player_id",
        ] + self.config.FEATURE_CONFIG["tracking_cols"]
        # Ensure unique columns in case config has duplicates
        tracking_cols = list(set(tracking_cols))

        df_tracking = pd.read_csv(tracking_path, usecols=lambda c: c in tracking_cols)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        # 5. Merge Tracking Data
        df_merged = self.merge_tracking_data(df_labels, df_tracking)

        # 6. Save Cache
        print(f"[{mode}] Saving merged data to {cache_path}...")
        df_merged.to_parquet(cache_path, index=False)

        # Cleanup
        del df_labels, df_tracking
        gc.collect()

        return reduce_mem_usage(df_merged)

    def merge_tracking_data(self, df_labels, df_tracking):
        """
        Merges tracking data onto the labels dataframe for both Player 1 and Player 2.

        Args:
            df_labels (pd.DataFrame): Dataframe containing contact labels and player pairs.
            df_tracking (pd.DataFrame): Dataframe containing player tracking info.

        Returns:
            pd.DataFrame: Merged dataframe with suffixes '_p1' and '_p2'.
        """
        print("Merging tracking data...")

        # Filter tracking data to only include game_plays present in labels to reduce join size
        relevant_plays = df_labels["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        # --- Merge Player 1 ---
        # Rename tracking columns for P1
        # We drop 'nfl_player_id' after merge or rename it during merge preparation
        # Strategy: Merge on [game_play, step, nfl_player_id]

        print("Merging Player 1 tracking...")
        df_merged = pd.merge(
            df_labels,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )

        # Rename the newly added tracking columns to have _p1 suffix
        # Identify columns that came from tracking (excluding join keys)
        tracking_feats = [
            c
            for c in df_tracking.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_map_p1 = {col: f"{col}_p1" for col in tracking_feats}
        df_merged.rename(columns=rename_map_p1, inplace=True)

        # Drop redundant nfl_player_id column from the merge
        if "nfl_player_id" in df_merged.columns:
            df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # --- Merge Player 2 ---
        print("Merging Player 2 tracking...")
        # Note: Player 2 can be 'G'. In this case, tracking data will be NaN, which is expected.

        df_merged = pd.merge(
            df_merged,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )

        # Rename tracking columns for P2
        rename_map_p2 = {col: f"{col}_p2" for col in tracking_feats}
        df_merged.rename(columns=rename_map_p2, inplace=True)

        # Drop redundant columns
        if "nfl_player_id" in df_merged.columns:
            df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # Clean up any potential duplicate columns ending in _drop (artifact of merges)
        drop_cols = [c for c in df_merged.columns if c.endswith("_drop")]
        if drop_cols:
            df_merged.drop(columns=drop_cols, inplace=True)

        return df_merged

    def load_helmets(self, mode="train"):
        """
        Loads helmet data. Helper function if visual features are needed later.

        Args:
            mode (str): 'train' or 'test'. Validation uses 'train'.

        Returns:
            pd.DataFrame: Helmet data.
        """
        if mode in ["train", "validation"]:
            path = self.config.PATH_CONFIG["train_helmets"]
        else:
            path = self.config.PATH_CONFIG["test_helmets"]

        print(f"[{mode}] Loading helmet data from {path}...")
        df = pd.read_csv(path)
        return reduce_mem_usage(df)
