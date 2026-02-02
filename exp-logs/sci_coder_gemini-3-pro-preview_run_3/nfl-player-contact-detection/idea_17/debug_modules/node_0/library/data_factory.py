import pandas as pd
import numpy as np
import os
from library.config import Config


class DataFactory:
    """
    Handles data ingestion, splitting, and caching for the NFL Contact Detection task.
    Distinguishes between Train, Validation, and Test modes to ensure correct raw data sources.
    """

    @staticmethod
    def load_dataset(mode="train", load_cached_data=True, debug=False, debug_size=1000):
        """
        Loads the metadata, tracking data, and helmet data for the specified mode.

        Args:
            mode (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): If True, attempts to load filtered data from parquet cache.
            debug (bool): If True, subsamples the metadata to process a smaller dataset.
            debug_size (int): Number of rows to sample if debug is True.

        Returns:
            tuple: (df_meta, df_tracking, df_helmets)
        """
        # 1. Determine Paths based on Mode
        if mode == "train":
            meta_path = Config.TRAIN_META_PATH
            tracking_source = Config.TRAIN_TRACKING_PATH
            helmets_source = Config.TRAIN_HELMETS_PATH
        elif mode == "validation":
            meta_path = Config.VAL_META_PATH
            # Validation comes from the training set split, so it uses train raw files
            tracking_source = Config.TRAIN_TRACKING_PATH
            helmets_source = Config.TRAIN_HELMETS_PATH
        elif mode == "test":
            meta_path = Config.TEST_META_PATH
            tracking_source = Config.TEST_TRACKING_PATH
            helmets_source = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'validation', or 'test'."
            )

        # 2. Load Metadata
        print(f"[{mode.upper()}] Loading metadata from {meta_path}...")
        df_meta = pd.read_csv(meta_path)

        # Debug Sampling
        if debug:
            print(f"[{mode.upper()}] Debug mode: Sampling {debug_size} rows...")
            df_meta = df_meta.sample(
                n=min(len(df_meta), debug_size), random_state=Config.SEED
            ).copy()

        # 3. Define Cache Paths
        # Cache key depends on mode and debug status
        debug_suffix = f"_debug{debug_size}" if debug else ""
        cache_tracking_path = os.path.join(
            Config.WORKING_DIR, f"tracking_{mode}{debug_suffix}.parquet"
        )
        cache_helmets_path = os.path.join(
            Config.WORKING_DIR, f"helmets_{mode}{debug_suffix}.parquet"
        )

        # 4. Load Tracking Data (Cached or Raw)
        df_tracking = None
        if load_cached_data and os.path.exists(cache_tracking_path):
            print(
                f"[{mode.upper()}] Loading tracking data from cache: {cache_tracking_path}"
            )
            df_tracking = pd.read_parquet(cache_tracking_path)
        else:
            print(
                f"[{mode.upper()}] Cache miss or force reload. Processing raw tracking data from {tracking_source}..."
            )
            # Load full raw file
            # Use low_memory=False to avoid mixed type warnings if chunks are processed, though here we read all.
            # Specifying dtypes can save memory but we stick to defaults for safety unless OOM occurs.
            raw_tracking = pd.read_csv(tracking_source)

            # Filter to relevant game_plays
            relevant_plays = df_meta["game_play"].unique()
            df_tracking = raw_tracking[
                raw_tracking["game_play"].isin(relevant_plays)
            ].copy()

            # Save to cache
            print(
                f"[{mode.upper()}] Saving filtered tracking data to {cache_tracking_path}..."
            )
            df_tracking.to_parquet(cache_tracking_path, index=False)

            # Clean up
            del raw_tracking

        # 5. Load Helmets Data (Cached or Raw)
        df_helmets = None
        if load_cached_data and os.path.exists(cache_helmets_path):
            print(
                f"[{mode.upper()}] Loading helmets data from cache: {cache_helmets_path}"
            )
            df_helmets = pd.read_parquet(cache_helmets_path)
        else:
            print(
                f"[{mode.upper()}] Cache miss or force reload. Processing raw helmets data from {helmets_source}..."
            )
            raw_helmets = pd.read_csv(helmets_source)

            # Filter to relevant game_plays
            relevant_plays = df_meta["game_play"].unique()
            df_helmets = raw_helmets[
                raw_helmets["game_play"].isin(relevant_plays)
            ].copy()

            # Save to cache
            print(
                f"[{mode.upper()}] Saving filtered helmets data to {cache_helmets_path}..."
            )
            df_helmets.to_parquet(cache_helmets_path, index=False)

            # Clean up
            del raw_helmets

        print(
            f"[{mode.upper()}] Data loaded. Meta: {df_meta.shape}, Tracking: {df_tracking.shape}, Helmets: {df_helmets.shape}"
        )
        return df_meta, df_tracking, df_helmets

    @staticmethod
    def split_contact_ids(df):
        """
        Splits the metadata/labels dataframe into two streams based on the contact type.

        Stream A: Interaction Model (Player-Player contact)
        Stream B: Hybrid Impact Model (Player-Ground contact)

        Args:
            df (pd.DataFrame): The dataframe containing 'nfl_player_id_2' column.

        Returns:
            tuple: (df_stream_a, df_stream_b)
        """
        # Ensure nfl_player_id_2 is string for consistent comparison
        # In some CSV reads, 'G' makes the col object, but if a chunk has no 'G', it might be int.
        # We force string conversion to be safe.
        player_2_col = df["nfl_player_id_2"].astype(str)

        # Stream B: Player 2 is 'G' (Ground)
        mask_ground = player_2_col == "G"

        df_stream_b = df[mask_ground].copy()
        df_stream_a = df[~mask_ground].copy()

        return df_stream_a, df_stream_b
