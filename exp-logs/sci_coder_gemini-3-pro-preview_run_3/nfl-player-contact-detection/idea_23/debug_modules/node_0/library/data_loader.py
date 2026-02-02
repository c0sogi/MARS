import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import generate_hash


class DataLoader:
    def __init__(self):
        """
        Initializes the DataLoader and ensures the cache directory exists.
        """
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_metadata(self, mode="train"):
        """
        Loads the metadata file for the specified mode.

        Args:
            mode (str): One of 'train', 'validation', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe.
        """
        if mode == "train":
            path = Config.TRAIN_META_PATH
        elif mode == "validation":
            path = Config.VAL_META_PATH
        elif mode == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        return pd.read_csv(path)

    def load_dataset(self, mode="train", load_cached_data=True, max_plays=None):
        """
        Loads metadata, tracking, and helmet data for the specified mode.
        Implements caching for tracking and helmet data to speed up subsequent runs.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): If True, attempts to load from parquet cache.
            max_plays (int, optional): If set, limits the number of unique plays loaded (for debugging).

        Returns:
            tuple: (df_meta, df_tracking, df_helmets)
        """
        # 1. Load Metadata
        df_meta = self.load_metadata(mode)

        # Handle debugging limit (max_plays)
        all_plays = df_meta["game_play"].unique()
        if max_plays is not None and max_plays < len(all_plays):
            # Sort to ensure deterministic subset
            relevant_plays = sorted(list(all_plays))[:max_plays]
            df_meta = df_meta[df_meta["game_play"].isin(relevant_plays)].copy()
        else:
            relevant_plays = sorted(list(all_plays))

        # 2. Determine Source Files
        # Validation split comes from the training source files
        if mode in ["train", "validation"]:
            tracking_source = Config.TRACKING_PATH_TRAIN
            helmets_source = Config.HELMETS_PATH_TRAIN
        elif mode == "test":
            tracking_source = Config.TRACKING_PATH_TEST
            helmets_source = Config.HELMETS_PATH_TEST
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 3. Load Tracking Data (with Caching)
        # Generate hash based on source, mode, and the specific subset of plays
        tracking_config = {
            "source": tracking_source,
            "mode": mode,
            "plays_subset_hash": generate_hash({"plays": relevant_plays}),
        }
        tracking_hash = generate_hash(tracking_config)
        tracking_cache_path = os.path.join(
            self.cache_dir, f"tracking_{mode}_{tracking_hash}.parquet"
        )

        df_tracking = None
        if load_cached_data and os.path.exists(tracking_cache_path):
            try:
                df_tracking = pd.read_parquet(tracking_cache_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load tracking cache ({e}). Reloading from source."
                )
                df_tracking = None

        if df_tracking is None:
            # Load raw CSV and filter
            # Reading full CSV is feasible with available RAM (220GB)
            df_raw = pd.read_csv(tracking_source)
            df_tracking = df_raw[df_raw["game_play"].isin(relevant_plays)].copy()

            # Save to cache
            df_tracking.to_parquet(tracking_cache_path, index=False)

        # 4. Load Helmets Data (with Caching)
        helmets_config = {
            "source": helmets_source,
            "mode": mode,
            "plays_subset_hash": generate_hash({"plays": relevant_plays}),
        }
        helmets_hash = generate_hash(helmets_config)
        helmets_cache_path = os.path.join(
            self.cache_dir, f"helmets_{mode}_{helmets_hash}.parquet"
        )

        df_helmets = None
        if load_cached_data and os.path.exists(helmets_cache_path):
            try:
                df_helmets = pd.read_parquet(helmets_cache_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load helmets cache ({e}). Reloading from source."
                )
                df_helmets = None

        if df_helmets is None:
            # Load raw CSV and filter
            df_raw = pd.read_csv(helmets_source)
            df_helmets = df_raw[df_raw["game_play"].isin(relevant_plays)].copy()

            # Save to cache
            df_helmets.to_parquet(helmets_cache_path, index=False)

        return df_meta, df_tracking, df_helmets

    def get_stream_data(self, df_labels):
        """
        Partitions the dataset into Stream A (Interaction) and Stream B (Impact).

        Stream A: Contact between two players.
        Stream B: Contact between a player and the ground (player2 == 'G').

        Args:
            df_labels (pd.DataFrame): The labels/metadata dataframe.

        Returns:
            tuple: (df_stream_a, df_stream_b)
        """
        # Ensure comparison is robust by converting to string (handles mixed types if any)
        p2_col = df_labels["nfl_player_id_2"].astype(str)

        # Identify Ground contacts
        mask_ground = p2_col == "G"

        df_stream_b = df_labels[mask_ground].copy()
        df_stream_a = df_labels[~mask_ground].copy()

        return df_stream_a, df_stream_b
