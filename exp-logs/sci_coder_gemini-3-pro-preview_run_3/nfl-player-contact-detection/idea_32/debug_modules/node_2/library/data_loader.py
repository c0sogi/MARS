import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_config_hash


class DataLoader:
    """
    Handles the loading, filtering, and caching of competition datasets.
    Implements strict mode-based file selection and ensures data consistency
    between labels, tracking data, and video metadata.
    """

    def load_data(self, mode: str, load_cached_data: bool = True, debug: bool = False):
        """
        Loads the dataset for a specific mode (train, validation, test).

        Args:
            mode (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to attempt loading from the parquet cache.
            debug (bool): If True, loads a small subset of data for debugging purposes.
                          Disables saving to cache to prevent corruption.

        Returns:
            tuple: (df_labels, df_tracking, df_helmets, df_video_meta)
        """
        if mode not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'validation', or 'test'."
            )

        # 1. Setup Cache Paths
        # We include 'debug' in the hash or filename to avoid mixing full and partial datasets
        config_hash = get_config_hash()
        debug_suffix = "_debug" if debug else ""

        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        cache_files = {
            "labels": os.path.join(
                cache_dir, f"data_{mode}_{config_hash}{debug_suffix}_labels.parquet"
            ),
            "tracking": os.path.join(
                cache_dir, f"data_{mode}_{config_hash}{debug_suffix}_tracking.parquet"
            ),
            "helmets": os.path.join(
                cache_dir, f"data_{mode}_{config_hash}{debug_suffix}_helmets.parquet"
            ),
            "video_meta": os.path.join(
                cache_dir, f"data_{mode}_{config_hash}{debug_suffix}_video_meta.parquet"
            ),
        }

        # 2. Try Loading from Cache
        if load_cached_data:
            all_exist = all(os.path.exists(path) for path in cache_files.values())
            if all_exist:
                print(f"[{mode.upper()}] Loading data from cache ({config_hash})...")
                try:
                    df_labels = pd.read_parquet(cache_files["labels"])
                    df_tracking = pd.read_parquet(cache_files["tracking"])
                    df_helmets = pd.read_parquet(cache_files["helmets"])
                    df_video_meta = pd.read_parquet(cache_files["video_meta"])
                    return df_labels, df_tracking, df_helmets, df_video_meta
                except Exception as e:
                    print(f"[{mode.upper()}] Cache load failed: {e}. Recomputing...")
            else:
                print(f"[{mode.upper()}] Cache miss. Loading raw data...")

        # 3. Determine Source Files based on Mode
        # Validation mode uses Train raw files but Validation metadata split
        if mode == "train":
            meta_path = Config.TRAIN_META_PATH
            tracking_path = Config.TRACKING_PATH_TRAIN
            helmets_path = Config.HELMETS_PATH_TRAIN
            video_meta_path = Config.VIDEO_META_TRAIN
        elif mode == "validation":
            meta_path = Config.VAL_META_PATH
            tracking_path = Config.TRACKING_PATH_TRAIN
            helmets_path = Config.HELMETS_PATH_TRAIN
            video_meta_path = Config.VIDEO_META_TRAIN
        else:  # test
            meta_path = Config.TEST_META_PATH
            tracking_path = Config.TRACKING_PATH_TEST
            helmets_path = Config.HELMETS_PATH_TEST
            video_meta_path = Config.VIDEO_META_TEST

        # 4. Load Labels / Metadata
        print(f"[{mode.upper()}] Loading metadata from {meta_path}...")
        df_labels = pd.read_csv(meta_path)

        # Apply Debug Slicing
        if debug:
            print(f"[{mode.upper()}] Debug mode: Slicing to top 500 rows/plays.")
            # Slice by game_play to ensure we get complete plays
            unique_plays = df_labels["game_play"].unique()[:5]  # Take 5 plays for debug
            df_labels = df_labels[df_labels["game_play"].isin(unique_plays)].copy()

        # Get relevant game_plays to filter the massive raw files
        relevant_game_plays = df_labels["game_play"].unique()
        print(
            f"[{mode.upper()}] Filtering for {len(relevant_game_plays)} unique plays."
        )

        # 5. Load and Filter Raw Data

        # Tracking Data
        print(f"[{mode.upper()}] Loading and filtering tracking data...")
        # Reading full CSV then filtering.
        # Note: For extremely large files, chunking would be better, but 1.2M rows fits in memory easily.
        df_tracking = pd.read_csv(tracking_path)
        df_tracking = df_tracking[
            df_tracking["game_play"].isin(relevant_game_plays)
        ].copy()

        # Helmets Data
        print(f"[{mode.upper()}] Loading and filtering helmets data...")
        df_helmets = pd.read_csv(helmets_path)
        df_helmets = df_helmets[
            df_helmets["game_play"].isin(relevant_game_plays)
        ].copy()

        # Video Metadata
        print(f"[{mode.upper()}] Loading and filtering video metadata...")
        df_video_meta = pd.read_csv(video_meta_path)
        df_video_meta = df_video_meta[
            df_video_meta["game_play"].isin(relevant_game_plays)
        ].copy()

        # 6. Save to Cache
        # We only save if not in debug mode (or if we implemented a specific debug cache, which we did via suffix)
        # However, typically we want to persist the result of the expensive load/filter op.
        print(f"[{mode.upper()}] Saving processed data to cache...")
        df_labels.to_parquet(cache_files["labels"], index=False)
        df_tracking.to_parquet(cache_files["tracking"], index=False)
        df_helmets.to_parquet(cache_files["helmets"], index=False)
        df_video_meta.to_parquet(cache_files["video_meta"], index=False)

        print(f"[{mode.upper()}] Data loading complete.")
        print(f"  Labels: {df_labels.shape}")
        print(f"  Tracking: {df_tracking.shape}")
        print(f"  Helmets: {df_helmets.shape}")

        return df_labels, df_tracking, df_helmets, df_video_meta
