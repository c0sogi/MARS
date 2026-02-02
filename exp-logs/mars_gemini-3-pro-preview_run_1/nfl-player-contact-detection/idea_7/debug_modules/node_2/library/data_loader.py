import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    WORKING_DIR,
)
from library.utils import setup_logger, CacheManager


class DataLoader:
    """
    Handles loading of metadata and tracking data, and creation of the base merged table.
    """

    def __init__(self, cache_dir=WORKING_DIR):
        self.logger = setup_logger()
        self.cache_manager = CacheManager(cache_dir)

    def load_metadata(self, mode="train"):
        """
        Loads the metadata CSV for the specified mode.
        """
        if mode == "train":
            path = TRAIN_METADATA_PATH
        elif mode == "val":
            path = VAL_METADATA_PATH
        elif mode == "test":
            path = TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        self.logger.info(f"Loading {mode} metadata from {path}...")
        return pd.read_csv(path)

    def load_tracking(self, mode="train"):
        """
        Loads the tracking data CSV appropriate for the specified mode.
        Train and Val modes share the training tracking data.
        """
        if mode in ["train", "val"]:
            path = TRAIN_TRACKING_PATH
        elif mode == "test":
            path = TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found at {path}")

        self.logger.info(f"Loading tracking data for {mode} from {path}...")
        return pd.read_csv(path)

    def prepare_base_table(self, mode="train", load_cached_data=True, n_rows=None):
        """
        Merges metadata with tracking data to create the base table.
        Uses caching to avoid re-computation.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            n_rows (int, optional): If provided, limits the input metadata to n_rows for debugging.

        Returns:
            pd.DataFrame: The merged base table.
        """
        # Define cache parameters
        cache_prefix = f"base_table_{mode}"
        cache_params = {"mode": mode, "version": "v1", "n_rows": n_rows}

        cache_path = self.cache_manager.get_cache_path(cache_prefix, cache_params)

        # 1. Try Load from Cache
        if load_cached_data:
            df = self.cache_manager.load(cache_path)
            if df is not None:
                self.logger.info(f"Loaded {mode} base table from cache: {cache_path}")
                return df

        self.logger.info(f"Computing base table for {mode} (n_rows={n_rows})...")

        # 2. Load Raw Data
        df_meta = self.load_metadata(mode)
        df_tracking = self.load_tracking(mode)

        # Apply debugging limit if requested
        if n_rows is not None and n_rows < len(df_meta):
            self.logger.info(f"Subsampling metadata to {n_rows} rows.")
            df_meta = df_meta.iloc[:n_rows].copy()

        # 3. Preprocess Tracking Data
        # Select relevant columns to minimize memory usage
        tracking_cols = [
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
            "position",
        ]

        # Ensure tracking columns exist (intersection with available columns)
        available_cols = [c for c in tracking_cols if c in df_tracking.columns]
        df_tracking = df_tracking[available_cols].copy()

        # Ensure join keys are consistent types
        df_tracking["game_play"] = df_tracking["game_play"].astype(str)
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

        # 4. Merge Player 1
        self.logger.info("Merging Player 1 tracking data...")
        df_merged = pd.merge(
            df_meta,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )

        # Rename P1 columns
        rename_map_p1 = {
            col: f"{col}_p1"
            for col in available_cols
            if col not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged.rename(columns=rename_map_p1, inplace=True)

        # Remove join key artifact if present
        if "nfl_player_id" in df_merged.columns:
            df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # 5. Merge Player 2
        self.logger.info("Merging Player 2 tracking data...")

        # Handle 'G' (Ground) in nfl_player_id_2
        # Create a numeric join key. 'G' becomes NaN.
        df_merged["nfl_player_id_2_join"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        # Merge
        df_merged = pd.merge(
            df_merged,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_2_join"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )

        # Rename P2 columns
        rename_map_p2 = {
            col: f"{col}_p2"
            for col in available_cols
            if col not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged.rename(columns=rename_map_p2, inplace=True)

        # Cleanup
        if "nfl_player_id" in df_merged.columns:
            df_merged.drop(columns=["nfl_player_id"], inplace=True)
        df_merged.drop(columns=["nfl_player_id_2_join"], inplace=True)

        # 6. Save to Cache
        self.logger.info(f"Saving {mode} base table to cache: {cache_path}")
        self.cache_manager.save(df_merged, cache_path)

        return df_merged
