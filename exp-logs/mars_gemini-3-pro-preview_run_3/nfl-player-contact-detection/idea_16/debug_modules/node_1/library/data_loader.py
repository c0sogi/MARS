import pandas as pd
import numpy as np
import os
from library.config import Config


class DataLoader:
    """
    Handles loading, filtering, and merging of NFL Player Contact Detection datasets.
    Implements caching to speed up iterative development.
    """

    def __init__(self, run_mode="train"):
        """
        Initialize the DataLoader.

        Args:
            run_mode (str): One of 'train', 'validation', 'test'.
        """
        self.run_mode = run_mode
        self.cache_dir = Config.WORKING_DIR

        # Define paths based on mode
        if run_mode == "train":
            self.meta_path = Config.TRAIN_META_PATH
            self.tracking_path = Config.TRAIN_TRACKING_PATH
            self.helmets_path = Config.TRAIN_HELMETS_PATH
        elif run_mode == "validation":
            self.meta_path = Config.VAL_META_PATH
            self.tracking_path = Config.TRAIN_TRACKING_PATH
            self.helmets_path = Config.TRAIN_HELMETS_PATH
        elif run_mode == "test":
            self.meta_path = Config.TEST_META_PATH
            self.tracking_path = Config.TEST_TRACKING_PATH
            self.helmets_path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid run_mode: {run_mode}")

    def load_metadata(self):
        """
        Loads the metadata CSV for the current run mode.

        Returns:
            pd.DataFrame: Metadata containing labels and video paths.
        """
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        df = pd.read_csv(self.meta_path)

        # Ensure ID columns are strings for consistent merging
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

        return df

    def load_tracking(self, relevant_game_plays):
        """
        Loads and filters player tracking data.

        Args:
            relevant_game_plays (set/list): List of game_play IDs to keep.

        Returns:
            pd.DataFrame: Filtered tracking data.
        """
        # Read tracking data
        # Using low_memory=False to prevent mixed type warnings if chunks are processed
        df_tracking = pd.read_csv(self.tracking_path, low_memory=False)

        # Filter by game_play
        df_tracking = df_tracking[
            df_tracking["game_play"].isin(relevant_game_plays)
        ].copy()

        # Ensure ID is string
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        return df_tracking

    def load_helmets(self, relevant_game_plays):
        """
        Loads and filters baseline helmet data.

        Args:
            relevant_game_plays (set/list): List of game_play IDs to keep.

        Returns:
            pd.DataFrame: Filtered helmet data.
        """
        df_helmets = pd.read_csv(self.helmets_path, low_memory=False)

        # Filter by game_play
        df_helmets = df_helmets[
            df_helmets["game_play"].isin(relevant_game_plays)
        ].copy()

        # Ensure ID is string
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

        return df_helmets

    def merge_data(self, df_labels, df_tracking, df_helmets, load_cached_data=True):
        """
        Merges labels with tracking and helmet data.
        Implements caching mechanism.

        Args:
            df_labels (pd.DataFrame): The base labels/metadata dataframe.
            df_tracking (pd.DataFrame): The tracking dataframe.
            df_helmets (pd.DataFrame): The helmets dataframe.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The merged dataframe ready for feature engineering.
        """
        cache_file = os.path.join(
            self.cache_dir, f"merged_data_{self.run_mode}.parquet"
        )

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading merged data from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Cache miss or force reload. Merging data for {self.run_mode}...")

        # --- Pre-processing for Merge ---

        # 1. Prepare Tracking
        # We need to merge tracking for Player 1 and Player 2 separately.
        # Rename columns to avoid collisions during merge.
        track_cols = [
            c
            for c in df_tracking.columns
            if c not in ["game_play", "step", "nfl_player_id", "datetime"]
        ]

        # 2. Prepare Helmets
        # Helmets are given by frame. We need to map step -> frame.
        # Formula: Frame = 300 (offset) + Step * 5.994 (approx)
        # We calculate this on the labels dataframe to create a join key.
        df_labels["frame_approx"] = (
            (300 + df_labels["step"] * 5.994).round().astype(int)
        )

        # Pivot helmets to have Sideline and Endzone on the same row for a (game_play, frame, player)
        # Columns of interest in helmets: game_play, frame, nfl_player_id, view, left, width, top, height
        helmet_features = ["left", "width", "top", "height"]

        # Split by view
        helmets_sideline = df_helmets[df_helmets["view"] == "Sideline"].copy()
        helmets_endzone = df_helmets[df_helmets["view"] == "Endzone"].copy()

        # Rename features
        rename_side = {c: f"view_sideline_{c}" for c in helmet_features}
        helmets_sideline = helmets_sideline.rename(columns=rename_side)

        rename_end = {c: f"view_endzone_{c}" for c in helmet_features}
        helmets_endzone = helmets_endzone.rename(columns=rename_end)

        # Select relevant columns for merge
        merge_cols_side = ["game_play", "frame", "nfl_player_id"] + list(
            rename_side.values()
        )
        merge_cols_end = ["game_play", "frame", "nfl_player_id"] + list(
            rename_end.values()
        )

        helmets_sideline = helmets_sideline[merge_cols_side]
        helmets_endzone = helmets_endzone[merge_cols_end]

        # --- Merging Player 1 ---
        print("Merging Player 1 data...")

        # Merge Tracking P1
        df_merged = pd.merge(
            df_labels,
            df_tracking.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge Helmets P1 (Sideline)
        df_merged = pd.merge(
            df_merged,
            helmets_sideline.add_suffix("_p1"),
            left_on=["game_play", "frame_approx", "nfl_player_id_1"],
            right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge Helmets P1 (Endzone)
        df_merged = pd.merge(
            df_merged,
            helmets_endzone.add_suffix("_p1"),
            left_on=["game_play", "frame_approx", "nfl_player_id_1"],
            right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
            how="left",
        )

        # --- Merging Player 2 ---
        print("Merging Player 2 data...")

        # Note: Player 2 can be 'G'. In that case, merges will result in NaNs, which is correct.

        # Merge Tracking P2
        df_merged = pd.merge(
            df_merged,
            df_tracking.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Merge Helmets P2 (Sideline)
        df_merged = pd.merge(
            df_merged,
            helmets_sideline.add_suffix("_p2"),
            left_on=["game_play", "frame_approx", "nfl_player_id_2"],
            right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Merge Helmets P2 (Endzone)
        df_merged = pd.merge(
            df_merged,
            helmets_endzone.add_suffix("_p2"),
            left_on=["game_play", "frame_approx", "nfl_player_id_2"],
            right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Clean up redundant join columns if desired, but keeping them for debug is fine.
        # We drop the specific join keys that were added with suffixes to save space
        drop_cols = [
            c
            for c in df_merged.columns
            if c.endswith("_p1")
            and (
                "game_play" in c or "step" in c or "frame" in c or "nfl_player_id" in c
            )
        ]
        drop_cols += [
            c
            for c in df_merged.columns
            if c.endswith("_p2")
            and (
                "game_play" in c or "step" in c or "frame" in c or "nfl_player_id" in c
            )
        ]
        # Keep original IDs

        df_merged.drop(columns=drop_cols, inplace=True, errors="ignore")

        # --- Save to Cache ---
        print(f"Saving merged data to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df_merged.to_parquet(cache_file, index=False)

        return df_merged
