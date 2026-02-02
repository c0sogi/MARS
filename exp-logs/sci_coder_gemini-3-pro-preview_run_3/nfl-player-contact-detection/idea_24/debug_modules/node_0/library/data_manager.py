import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, get_config_hash


class DataManager:
    """
    Handles the loading, alignment, and caching of Tracking, Helmet, and Metadata datasets.
    """

    def __init__(self):
        self.logger = setup_logger("DataManager")
        self.config_hash = get_config_hash()
        self.working_dir = Config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_source_split(self, mode):
        """
        Determines the source file split ('train' or 'test') based on the mode.
        'validation' mode uses 'train' source files.
        """
        if mode in ["train", "validation"]:
            return "train"
        elif mode == "test":
            return "test"
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'validation', or 'test'."
            )

    def load_metadata(self, mode="train"):
        """
        Loads the specific metadata file for the requested mode.
        """
        self.logger.info(f"Loading metadata for mode: {mode}")

        if mode == "train":
            path = Config.TRAIN_META
        elif mode == "validation":
            path = Config.VAL_META
        elif mode == "test":
            path = Config.TEST_META
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)
        self.logger.info(f"Loaded {len(df)} rows from {path}")
        return df

    def load_tracking(self, mode="train", metadata_df=None, load_cached_data=True):
        """
        Loads player tracking data.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            metadata_df (pd.DataFrame): The loaded metadata to filter valid game_plays.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Filtered tracking data.
        """
        split = self._get_source_split(mode)
        cache_filename = f"tracking_{mode}_{self.config_hash}.parquet"
        cache_path = os.path.join(self.working_dir, cache_filename)

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached tracking data from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Process from Scratch
        self.logger.info(f"Processing tracking data for mode: {mode} (Source: {split})")

        # Determine source path
        source_path = (
            Config.TRACKING_TRAIN if split == "train" else Config.TRACKING_TEST
        )

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Tracking source file not found: {source_path}")

        # Load raw data
        # Using pandas engine for CSV reading
        df_tracking = pd.read_csv(source_path)

        # Filter by game_play if metadata is provided
        if metadata_df is not None:
            valid_plays = metadata_df["game_play"].unique()
            original_count = len(df_tracking)
            df_tracking = df_tracking[df_tracking["game_play"].isin(valid_plays)].copy()
            self.logger.info(
                f"Filtered tracking data: {original_count} -> {len(df_tracking)} rows"
            )

        # Parse datetime
        if "datetime" in df_tracking.columns:
            df_tracking["datetime"] = pd.to_datetime(df_tracking["datetime"], utc=True)

        # Save to cache
        self.logger.info(f"Saving tracking data to cache: {cache_path}")
        df_tracking.to_parquet(cache_path, index=False)

        return df_tracking

    def load_helmets(self, mode="train", metadata_df=None, load_cached_data=True):
        """
        Loads helmet baseline data and merges it with video metadata to establish timestamps.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            metadata_df (pd.DataFrame): The loaded metadata to filter valid game_plays.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Filtered and enriched helmet data.
        """
        split = self._get_source_split(mode)
        cache_filename = f"helmets_{mode}_{self.config_hash}.parquet"
        cache_path = os.path.join(self.working_dir, cache_filename)

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached helmet data from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Process from Scratch
        self.logger.info(f"Processing helmet data for mode: {mode} (Source: {split})")

        # Determine source paths
        helmet_source_path = (
            Config.HELMETS_TRAIN if split == "train" else Config.HELMETS_TEST
        )
        video_meta_path = os.path.join(Config.INPUT_DIR, f"{split}_video_metadata.csv")

        if not os.path.exists(helmet_source_path):
            raise FileNotFoundError(
                f"Helmet source file not found: {helmet_source_path}"
            )
        if not os.path.exists(video_meta_path):
            raise FileNotFoundError(f"Video metadata file not found: {video_meta_path}")

        # Load raw datasets
        df_helmets = pd.read_csv(helmet_source_path)
        df_video_meta = pd.read_csv(video_meta_path)

        # Filter by game_play if metadata is provided
        if metadata_df is not None:
            valid_plays = metadata_df["game_play"].unique()
            original_count = len(df_helmets)
            df_helmets = df_helmets[df_helmets["game_play"].isin(valid_plays)].copy()
            self.logger.info(
                f"Filtered helmet data: {original_count} -> {len(df_helmets)} rows"
            )

            # Also filter video meta for efficiency
            df_video_meta = df_video_meta[
                df_video_meta["game_play"].isin(valid_plays)
            ].copy()

        # Merge Video Metadata to get start_time
        # Helmets: game_play, view, frame, ...
        # Video Meta: game_play, view, start_time, ...

        # Ensure view columns match (usually 'Sideline'/'Endzone')
        df_merged = pd.merge(
            df_helmets,
            df_video_meta[["game_play", "view", "start_time", "snap_time"]],
            on=["game_play", "view"],
            how="left",
        )

        # Calculate estimated datetime for each frame
        # Frame rate is 59.94 Hz
        # datetime = start_time + (frame - 1) / 59.94
        if "start_time" in df_merged.columns:
            start_times = pd.to_datetime(df_merged["start_time"], utc=True)
            frame_offsets = pd.to_timedelta((df_merged["frame"] - 1) / 59.94, unit="s")
            df_merged["datetime"] = start_times + frame_offsets

            # Drop raw string time columns to save space, keep datetime object
            df_merged.drop(columns=["start_time"], inplace=True)

        # Save to cache
        self.logger.info(f"Saving helmet data to cache: {cache_path}")
        df_merged.to_parquet(cache_path, index=False)

        return df_merged

    def load_video_metadata(self, mode="train"):
        """
        Loads the raw video metadata file for the corresponding split.
        """
        split = self._get_source_split(mode)
        path = os.path.join(Config.INPUT_DIR, f"{split}_video_metadata.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Video metadata file not found: {path}")

        return pd.read_csv(path)
