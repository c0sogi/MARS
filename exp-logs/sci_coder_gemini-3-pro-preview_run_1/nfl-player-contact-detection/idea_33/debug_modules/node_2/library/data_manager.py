import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import CacheManager, reduce_mem_usage


class DataManager:
    """
    Handles data ingestion, merging, and preliminary preprocessing for the DEIB-AME pipeline.
    Implements caching and the Sentinel Value Strategy for Ground interactions.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        # Columns to select from tracking data
        self.tracking_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "distance",  # Distance traveled since last frame
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]

    def load_train_data(self, load_cached_data=True, sample_size=None):
        """
        Loads and merges training data.
        """
        df = self._load_and_merge(
            metadata_path=Config.TRAIN_METADATA_PATH,
            tracking_path=Config.TRAIN_TRACKING_PATH,
            cache_filename="merged_train.parquet",
            load_cached_data=load_cached_data,
        )
        if sample_size and sample_size < len(df):
            return df.head(sample_size)
        return df

    def load_val_data(self, load_cached_data=True, sample_size=None):
        """
        Loads and merges validation data.
        """
        df = self._load_and_merge(
            metadata_path=Config.VAL_METADATA_PATH,
            tracking_path=Config.TRAIN_TRACKING_PATH,  # Validation uses train tracking file
            cache_filename="merged_val.parquet",
            load_cached_data=load_cached_data,
        )
        if sample_size and sample_size < len(df):
            return df.head(sample_size)
        return df

    def load_test_data(self, load_cached_data=True, sample_size=None):
        """
        Loads and merges test data.
        """
        df = self._load_and_merge(
            metadata_path=Config.TEST_METADATA_PATH,
            tracking_path=Config.TEST_TRACKING_PATH,
            cache_filename="merged_test.parquet",
            load_cached_data=load_cached_data,
        )
        if sample_size and sample_size < len(df):
            return df.head(sample_size)
        return df

    def _load_and_merge(
        self, metadata_path, tracking_path, cache_filename, load_cached_data
    ):
        """
        Core logic to load CSVs, merge tracking data for both players, apply sentinel values,
        and cache the result.
        """
        # 1. Check Cache
        if load_cached_data and self.cache_manager.exists(cache_filename):
            print(f"Loading cached merged data from {cache_filename}...")
            return self.cache_manager.load(cache_filename)

        print(f"Generating merged data for {cache_filename}...")

        # 2. Load Raw Data
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not os.path.exists(tracking_path):
            raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

        df_meta = pd.read_csv(metadata_path)
        df_tracking = pd.read_csv(tracking_path)

        # Optimize memory immediately
        df_meta = reduce_mem_usage(df_meta)
        df_tracking = reduce_mem_usage(df_tracking)

        # Filter tracking columns to keep only what's needed
        available_track_cols = [
            c for c in self.tracking_cols if c in df_tracking.columns
        ]
        df_tracking = df_tracking[available_track_cols]

        # 3. Merge Player 1 Tracking
        # Player 1 is always an integer ID
        print("Merging Player 1 tracking data...")
        df_merged = df_meta.merge(
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P1 columns and drop redundant id
        rename_p1 = {
            c: f"{c}_p1"
            for c in available_track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p1)
        df_merged = df_merged.drop(columns=["nfl_player_id"], errors="ignore")

        # 4. Merge Player 2 Tracking
        # Player 2 can be 'G' (Ground) or an integer ID.
        # We coerce to numeric; 'G' becomes NaN and won't match, which is desired.
        print("Merging Player 2 tracking data...")
        df_merged["nfl_player_id_2_int"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = df_merged.merge(
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_2_int"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P2 columns and drop redundant ids
        rename_p2 = {
            c: f"{c}_p2"
            for c in available_track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p2)
        df_merged = df_merged.drop(
            columns=["nfl_player_id", "nfl_player_id_2_int"], errors="ignore"
        )

        # 5. Apply Sentinel Value Strategy for Distance
        print("Applying Sentinel Value Strategy...")

        # Ensure coordinates are floats for calculation
        coord_cols = [
            "x_position_p1",
            "y_position_p1",
            "x_position_p2",
            "y_position_p2",
        ]
        for col in coord_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].astype(float)

        # Calculate Euclidean distance between entities
        # If P2 is missing (or 'G'), this results in NaN initially
        if (
            "x_position_p1" in df_merged.columns
            and "x_position_p2" in df_merged.columns
        ):
            dx = df_merged["x_position_p1"] - df_merged["x_position_p2"]
            dy = df_merged["y_position_p1"] - df_merged["y_position_p2"]
            df_merged["distance"] = np.sqrt(dx**2 + dy**2)
        else:
            df_merged["distance"] = np.nan

        # Explicitly set distance to Sentinel Value (-1.0) for Ground interactions
        ground_mask = df_merged["nfl_player_id_2"] == "G"
        df_merged.loc[ground_mask, "distance"] = Config.GROUND_DISTANCE_SENTINEL

        # 6. Final Cleanup
        df_merged = reduce_mem_usage(df_merged)

        # Save to cache
        self.cache_manager.save(df_merged, cache_filename)

        return df_merged
