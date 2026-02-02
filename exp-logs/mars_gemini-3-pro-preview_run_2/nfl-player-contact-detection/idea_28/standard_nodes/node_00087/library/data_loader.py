import os
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    """
    Handles the ingestion of raw CSV files and metadata for the TD-SRN solution.
    Implements efficient loading, filtering, and caching mechanisms.
    """

    @staticmethod
    def load_metadata(split: str = "train") -> pd.DataFrame:
        """
        Loads the ground truth labels or submission file for the specified split.

        Args:
            split (str): One of 'train', 'validation', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe containing contact_ids, video paths, etc.
        """
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at {path}")

        df = pd.read_csv(path)

        # Ensure game_play is string to match tracking data format
        if "game_play" in df.columns:
            df["game_play"] = df["game_play"].astype(str)

        return df

    @staticmethod
    def load_tracking_data(
        split: str, game_plays: list = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads player tracking data, optionally filtering by game_plays.
        Implements caching to Parquet for faster subsequent loads.

        Args:
            split (str): 'train', 'validation', or 'test'.
            game_plays (list, optional): List of game_play IDs to filter by.
                                         If None, loads all data for the source file.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The tracking data.
        """
        # Determine cache path
        cache_filename = f"tracking_{split}.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(
                    f"Failed to load cache from {cache_path}: {e}. Reloading from source."
                )

        # 2. Determine source path
        # Validation split uses the train source file but is a subset of plays
        if split in ["train", "validation"]:
            source_path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            source_path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Tracking source file not found at {source_path}")

        # 3. Load and Filter
        # Reading large CSVs can be slow, so we read and filter immediately
        # Using pyarrow engine for speed if available, else default
        try:
            df = pd.read_csv(source_path, engine="c", dtype={"game_play": str})
        except ValueError:
            df = pd.read_csv(source_path, dtype={"game_play": str})

        if game_plays is not None:
            # Ensure game_plays are strings for comparison
            game_plays_set = set(str(gp) for gp in game_plays)
            df = df[df["game_play"].isin(game_plays_set)].copy()

        # 4. Save to cache if deterministic processing (filtering) occurred
        # We ensure the directory exists as per requirements
        Config.setup_directories()

        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return df

    @staticmethod
    def load_helmets_data(
        split: str, game_plays: list = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads baseline helmet detection data, optionally filtering by game_plays.
        Implements caching to Parquet.

        Args:
            split (str): 'train', 'validation', or 'test'.
            game_plays (list, optional): List of game_play IDs to filter by.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The helmet data.
        """
        # Determine cache path
        cache_filename = f"helmets_{split}.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(
                    f"Failed to load cache from {cache_path}: {e}. Reloading from source."
                )

        # 2. Determine source path
        if split in ["train", "validation"]:
            source_path = Config.TRAIN_HELMETS_PATH
        elif split == "test":
            source_path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Helmets source file not found at {source_path}")

        # 3. Load and Filter
        # Helmets file is also large
        try:
            df = pd.read_csv(source_path, engine="c", dtype={"game_play": str})
        except ValueError:
            df = pd.read_csv(source_path, dtype={"game_play": str})

        if game_plays is not None:
            game_plays_set = set(str(gp) for gp in game_plays)
            df = df[df["game_play"].isin(game_plays_set)].copy()

        # 4. Save to cache
        Config.setup_directories()

        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

        return df
