import pandas as pd
import os
import numpy as np
from library.config import Config
from library.utils import generate_config_hash


class DataLoader:
    """
    Handles loading of metadata, tracking data, and helmet data with caching and split-aware filtering.
    """

    @staticmethod
    def load_metadata(split: str) -> pd.DataFrame:
        """
        Loads the metadata (labels) for a specific split.

        Args:
            split (str): One of 'train', 'validation', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe containing labels and video paths.
        """
        if split == "train":
            path = Config.TRAIN_META_PATH
        elif split == "validation":
            path = Config.VAL_META_PATH
        elif split == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', or 'test'."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Ensure datetime is parsed
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        return df

    @staticmethod
    def load_tracking(split: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads player tracking data for the specific split.

        Logic:
        - 'train' and 'validation' splits read from train_player_tracking.csv
        - 'test' split reads from test_player_tracking.csv
        - Data is filtered to only include game_plays present in the corresponding metadata split.
        - Caching is implemented using Parquet.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Filtered tracking data.
        """
        # 1. Determine Cache Path
        cache_filename = f"tracking_{split}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 2. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached tracking data for {split} from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing tracking data for {split} (Cache miss or force reload)...")

        # 3. Determine Source File
        if split in ["train", "validation"]:
            source_path = Config.TRAIN_TRACKING_PATH
        elif split == "test":
            source_path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source tracking file not found: {source_path}")

        # 4. Load Metadata to get valid game_plays for filtering
        meta_df = DataLoader.load_metadata(split)
        valid_game_plays = meta_df["game_play"].unique()

        # 5. Load and Filter Raw Data
        # Using chunks isn't strictly necessary given 220GB RAM, but good practice.
        # Here we load full then filter for simplicity given memory abundance.
        df_tracking = pd.read_csv(source_path)

        # Filter
        initial_shape = df_tracking.shape
        df_tracking = df_tracking[
            df_tracking["game_play"].isin(valid_game_plays)
        ].copy()
        print(
            f"Filtered tracking data for {split}: {initial_shape} -> {df_tracking.shape}"
        )

        # 6. Type Conversion
        if "datetime" in df_tracking.columns:
            df_tracking["datetime"] = pd.to_datetime(df_tracking["datetime"], utc=True)

        # 7. Save to Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_tracking.to_parquet(cache_path, index=False)
        print(f"Saved tracking cache to {cache_path}")

        return df_tracking

    @staticmethod
    def load_helmets(split: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads baseline helmet detection data for the specific split.

        Logic:
        - 'train' and 'validation' splits read from train_baseline_helmets.csv
        - 'test' split reads from test_baseline_helmets.csv
        - Data is filtered to only include game_plays present in the corresponding metadata split.
        - Caching is implemented using Parquet.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Filtered helmet data.
        """
        # 1. Determine Cache Path
        cache_filename = f"helmets_{split}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 2. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached helmet data for {split} from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Processing helmet data for {split} (Cache miss or force reload)...")

        # 3. Determine Source File
        if split in ["train", "validation"]:
            source_path = Config.TRAIN_HELMETS_PATH
        elif split == "test":
            source_path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source helmet file not found: {source_path}")

        # 4. Load Metadata to get valid game_plays for filtering
        meta_df = DataLoader.load_metadata(split)
        valid_game_plays = meta_df["game_play"].unique()

        # 5. Load and Filter Raw Data
        df_helmets = pd.read_csv(source_path)

        # Filter
        initial_shape = df_helmets.shape
        df_helmets = df_helmets[df_helmets["game_play"].isin(valid_game_plays)].copy()
        print(
            f"Filtered helmet data for {split}: {initial_shape} -> {df_helmets.shape}"
        )

        # 6. Save to Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_helmets.to_parquet(cache_path, index=False)
        print(f"Saved helmet cache to {cache_path}")

        return df_helmets
