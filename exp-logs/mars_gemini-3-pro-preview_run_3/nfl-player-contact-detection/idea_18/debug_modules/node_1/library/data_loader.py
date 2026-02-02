import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import get_hash


class DataLoader:
    """
    Handles loading, merging, and caching of NFL Contact Detection datasets.
    """

    def __init__(self):
        self.config = Config

    def load_metadata(self, mode):
        """
        Loads the metadata file for the specified mode (train/validation/test).
        """
        if mode == "train":
            path = self.config.TRAIN_META_PATH
        elif mode == "validation":
            path = self.config.VAL_META_PATH
        elif mode == "test":
            path = self.config.TEST_META_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        return pd.read_csv(path)

    def load_tracking(self, mode):
        """
        Loads the player tracking data.
        Validation mode uses the training tracking file.
        """
        if mode in ["train", "validation"]:
            path = self.config.TRAIN_TRACKING_PATH
        elif mode == "test":
            path = self.config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found: {path}")

        return pd.read_csv(path)

    def load_helmets(self, mode):
        """
        Loads the baseline helmet detection data.
        Validation mode uses the training helmet file.
        """
        if mode in ["train", "validation"]:
            path = self.config.TRAIN_HELMETS_PATH
        elif mode == "test":
            path = self.config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Helmets file not found: {path}")

        return pd.read_csv(path)

    def merge_data(self, df_labels, df_tracking, df_helmets):
        """
        Merges labels with tracking and helmet data for both players involved.

        Args:
            df_labels: DataFrame containing contact labels and pairs.
            df_tracking: DataFrame containing player tracking data.
            df_helmets: DataFrame containing helmet bounding boxes.

        Returns:
            DataFrame: A single merged dataframe with features for P1 and P2.
        """
        # --- 1. Preprocessing & Type Standardization ---
        # Ensure IDs are strings to handle 'G' (Ground) and numeric IDs consistently
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)

        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

        # Filter auxiliary data to relevant game_plays to optimize memory usage
        relevant_plays = df_labels["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # --- 2. Prepare Tracking Data ---
        # Select relevant columns
        tracking_cols = [
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
        # Only keep columns that actually exist in the source
        tracking_cols = [c for c in tracking_cols if c in df_tracking.columns]
        df_track_subset = df_tracking[tracking_cols].copy()

        # --- 3. Prepare Helmet Data ---
        # Calculate approximate frame number from step.
        # Video is 59.94Hz, Data is 10Hz. Snap is at 5s (frame 300).
        # Formula: frame = 300 + step * 6
        df_labels["frame"] = (300 + df_labels["step"] * 6).astype(int)

        # Pivot helmets to flatten 'Sideline' and 'Endzone' views into columns
        # We want one row per (game_play, frame, player)
        helmet_cols = [
            "game_play",
            "frame",
            "nfl_player_id",
            "view",
            "left",
            "width",
            "top",
            "height",
        ]
        df_helmets_subset = df_helmets[helmet_cols].copy()

        # Remove duplicates if any (though baseline file usually unique per view)
        df_helmets_subset = df_helmets_subset.drop_duplicates(
            subset=["game_play", "frame", "nfl_player_id", "view"]
        )

        df_helmets_pivoted = df_helmets_subset.pivot(
            index=["game_play", "frame", "nfl_player_id"],
            columns="view",
            values=["left", "width", "top", "height"],
        )

        # Flatten the MultiIndex columns (e.g., ('left', 'Sideline') -> 'left_Sideline')
        df_helmets_pivoted.columns = [
            f"{col[0]}_{col[1]}" for col in df_helmets_pivoted.columns
        ]
        df_helmets_pivoted = df_helmets_pivoted.reset_index()

        # --- 4. Merge Player 1 Data ---
        # Join Tracking P1
        df_merged = pd.merge(
            df_labels,
            df_track_subset.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Join Helmets P1
        df_merged = pd.merge(
            df_merged,
            df_helmets_pivoted.add_suffix("_p1"),
            left_on=["game_play", "frame", "nfl_player_id_1"],
            right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
            how="left",
        )

        # --- 5. Merge Player 2 Data ---
        # Join Tracking P2
        df_merged = pd.merge(
            df_merged,
            df_track_subset.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Join Helmets P2
        df_merged = pd.merge(
            df_merged,
            df_helmets_pivoted.add_suffix("_p2"),
            left_on=["game_play", "frame", "nfl_player_id_2"],
            right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
            how="left",
        )

        # --- 6. Cleanup ---
        # Remove redundant join keys that were added with suffixes
        redundant_suffixes = ["_p1", "_p2"]
        redundant_keys = ["game_play", "step", "nfl_player_id", "frame"]

        cols_to_drop = []
        for suffix in redundant_suffixes:
            for key in redundant_keys:
                col_name = key + suffix
                if col_name in df_merged.columns:
                    cols_to_drop.append(col_name)

        df_merged.drop(columns=cols_to_drop, inplace=True)

        return df_merged

    def get_dataset(self, mode, load_cached_data=True):
        """
        Main entry point. Loads, merges, and returns the dataset for the given mode.
        Implements caching to Parquet.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The merged dataset.
        """
        # Construct cache path
        cache_filename = f"merged_data_{mode}.parquet"
        cache_path = os.path.join(self.config.CACHE_DIR, cache_filename)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Generate from Scratch
        print(f"Generating {mode} data from scratch...")

        df_labels = self.load_metadata(mode)
        df_tracking = self.load_tracking(mode)
        df_helmets = self.load_helmets(mode)

        df_merged = self.merge_data(df_labels, df_tracking, df_helmets)

        # 3. Save Cache
        print(f"Saving {mode} data to {cache_path}...")
        # Ensure directory exists (redundant if Config handles it, but safe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_merged.to_parquet(cache_path, index=False)

        return df_merged
