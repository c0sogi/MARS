import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    SENTINEL_VALUE,
    SEED,
    IDEA_DIR,
)
from library.utils import CacheManager, setup_logger


class DataProcessor:
    """
    Handles data ingestion, merging of tracking data, and initial preprocessing
    for the RKS-MTE strategy.
    """

    def __init__(self):
        self.logger = setup_logger("data_processor")
        self.cache_manager = CacheManager(cache_dir=IDEA_DIR)

    def load_and_merge_data(
        self, split="train", load_cached_data=True, sample_size=None
    ):
        """
        Loads metadata and tracking data, merges them, and applies sentinel values
        for ground interactions.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            sample_size (int, optional): Number of rows to sample from metadata for debugging.

        Returns:
            pd.DataFrame: The merged dataframe with tracking features and distance.
        """
        # 1. Cache Check
        cache_params = {
            "split": split,
            "sample_size": sample_size,
            "stage": "merged_data",
        }

        if load_cached_data:
            cached_df = self.cache_manager.load(f"data_{split}", cache_params)
            if cached_df is not None:
                self.logger.info(
                    f"Loaded {split} data from cache with shape {cached_df.shape}"
                )
                return cached_df

        self.logger.info(f"Processing {split} data from scratch...")

        # 2. Determine File Paths
        if split == "train":
            meta_path = TRAIN_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH
        elif split == "val":
            meta_path = VAL_METADATA_PATH
            track_path = TRAIN_TRACKING_PATH  # Validation comes from train source
        elif split == "test":
            meta_path = TEST_METADATA_PATH
            track_path = TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # 3. Load Metadata
        self.logger.info(f"Loading metadata from {meta_path}")
        df_meta = pd.read_csv(meta_path)

        # Apply sampling if requested
        if sample_size is not None and sample_size < len(df_meta):
            self.logger.info(f"Sampling {sample_size} rows from metadata...")
            df_meta = df_meta.sample(n=sample_size, random_state=SEED).reset_index(
                drop=True
            )

        # 4. Load Tracking Data
        self.logger.info(f"Loading tracking data from {track_path}")
        df_track = pd.read_csv(track_path)

        # Ensure join keys have consistent types
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)

        df_track["game_play"] = df_track["game_play"].astype(str)
        df_track["step"] = df_track["step"].astype(int)

        # Tracking columns to keep and rename
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
        df_track = df_track[track_cols]

        # 5. Merge Player 1 Tracking
        self.logger.info("Merging Player 1 tracking data...")
        df_merged = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P1 columns
        rename_p1 = {
            c: f"{c}_p1"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p1)
        df_merged = df_merged.drop(columns=["nfl_player_id"])  # Drop redundant join key

        # 6. Merge Player 2 Tracking
        self.logger.info("Merging Player 2 tracking data...")

        # Handle 'G' in nfl_player_id_2 for merging
        # Convert to numeric, coercing errors ('G') to NaN so they don't match any player ID
        df_merged["nfl_player_id_2_join"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = pd.merge(
            df_merged,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_2_join"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P2 columns
        rename_p2 = {
            c: f"{c}_p2"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p2)
        df_merged = df_merged.drop(columns=["nfl_player_id", "nfl_player_id_2_join"])

        # Free up memory
        del df_track
        gc.collect()

        # 7. Feature Calculation & Sentinel Application
        self.logger.info("Calculating initial distance and applying sentinel values...")

        # Calculate Euclidean distance
        # Note: This will be NaN for Ground interactions (because x_position_p2 is NaN)
        df_merged["distance"] = np.sqrt(
            (df_merged["x_position_p1"] - df_merged["x_position_p2"]) ** 2
            + (df_merged["y_position_p1"] - df_merged["y_position_p2"]) ** 2
        )

        # Apply Sentinel Value Strategy for Ground
        # Explicitly set distance to SENTINEL_VALUE where Player 2 is Ground
        ground_mask = df_merged["nfl_player_id_2"] == "G"
        df_merged.loc[ground_mask, "distance"] = SENTINEL_VALUE

        # 8. Save to Cache
        self.logger.info(f"Saving merged {split} data to cache...")
        self.cache_manager.save(df_merged, f"data_{split}", cache_params)

        self.logger.info(f"Completed processing {split}. Shape: {df_merged.shape}")
        return df_merged
