import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import seed_everything


class DataManager:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)
        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _load_metadata(self, mode: str) -> pd.DataFrame:
        """
        Loads the specific metadata file based on mode.
        """
        if mode == "train":
            path = self.config.TRAIN_META_PATH
        elif mode == "validation":
            path = self.config.VAL_META_PATH
        elif mode == "test":
            path = self.config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Apply debug sampling if configured and in training mode
        if mode == "train" and self.config.DEBUG_SAMPLE_SIZE is not None:
            print(
                f"DEBUG: Sampling {self.config.DEBUG_SAMPLE_SIZE} rows from metadata."
            )
            df = df.head(int(self.config.DEBUG_SAMPLE_SIZE))

        return df

    def _load_tracking(self, mode: str) -> pd.DataFrame:
        """
        Loads the appropriate tracking data.
        Note: Validation uses TRAIN tracking data.
        """
        if mode in ["train", "validation"]:
            path = self.config.TRAIN_TRACKING_PATH
        elif mode == "test":
            path = self.config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found: {path}")

        # Load tracking data
        # We can optimize by specifying types if needed, but pandas inference is usually okay for this size
        df = pd.read_csv(path)
        return df

    def _merge_tracking_data(
        self, df_labels: pd.DataFrame, df_tracking: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges tracking data onto the labels for both Player 1 and Player 2.
        """
        # 1. Filter tracking data to relevant game_plays to reduce memory/time
        relevant_plays = df_labels["game_play"].unique()
        df_track_filtered = df_tracking[
            df_tracking["game_play"].isin(relevant_plays)
        ].copy()

        # 2. Prepare join keys
        # Ensure IDs are strings for consistent merging (handling 'G' logic implicitly via type)
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)

        df_track_filtered["nfl_player_id"] = df_track_filtered["nfl_player_id"].astype(
            str
        )

        # 3. Select relevant tracking columns to merge
        # We need position, speed, accel, orientation, direction
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "sa",
        ]
        df_track_subset = df_track_filtered[track_cols]

        # 4. Merge Player 1
        # Inner join because Player 1 must exist in tracking (it's a player)
        # Rename columns with suffix _p1
        df_merged = pd.merge(
            df_labels,
            df_track_subset.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",  # Use left to keep label rows even if tracking is missing (rare but possible)
        )

        # Drop redundant join keys
        df_merged = df_merged.drop(
            columns=["game_play_p1", "step_p1", "nfl_player_id_p1"]
        )

        # 5. Merge Player 2
        # Left join because Player 2 might be 'G' (Ground) or missing
        # Rename columns with suffix _p2
        df_merged = pd.merge(
            df_merged,
            df_track_subset.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Drop redundant join keys
        df_merged = df_merged.drop(
            columns=["game_play_p2", "step_p2", "nfl_player_id_p2"]
        )

        # Clean up
        del df_track_filtered
        del df_track_subset
        gc.collect()

        return df_merged

    def get_data(self, mode: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Main method to retrieve data. Handles caching.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The merged dataframe.
        """
        cache_filename = f"merged_data_{mode}.parquet"

        # Append debug suffix if in debug mode to avoid overwriting full cache
        if mode == "train" and self.config.DEBUG_SAMPLE_SIZE is not None:
            cache_filename = (
                f"merged_data_{mode}_debug_{self.config.DEBUG_SAMPLE_SIZE}.parquet"
            )

        cache_path = os.path.join(self.config.WORKING_DIR, cache_filename)

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        print(f"Processing raw data for {mode}...")

        # Load raw files
        df_meta = self._load_metadata(mode)
        df_tracking = self._load_tracking(mode)

        # Merge
        print(f"Merging tracking data for {mode} ({len(df_meta)} labels)...")
        df_merged = self._merge_tracking_data(df_meta, df_tracking)

        # Save to cache
        print(f"Saving {mode} data to cache: {cache_path}")
        df_merged.to_parquet(cache_path, index=False)

        return df_merged
