import os
import pandas as pd
from library.config import Config
from library.geometry_loader import extract_geometry_features


class DataManager:
    def __init__(self):
        """
        Initializes the DataManager and ensures the working directory exists.
        """
        # Ensure working directory exists for caching
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def _load_dataset(
        self, metadata_path, cache_path, sample_size=None, load_cached_data=True
    ):
        """
        Generic helper method to load a dataset (train, val, or test).

        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_path (str): Path to save/load the parquet cache file for geometry features.
            sample_size (int, optional): Number of samples to load (for debugging purposes).
            load_cached_data (bool): Whether to attempt loading geometry features from cache.

        Returns:
            pd.DataFrame: The loaded dataframe with metadata and geometry features merged.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        # 1. Load metadata
        df = pd.read_csv(metadata_path)

        # 2. Apply sampling if requested
        if sample_size is not None:
            # Slice the dataframe to the requested size
            df = df.iloc[:sample_size].copy()

            # Modify cache path to avoid overwriting the full dataset cache with a partial one
            # Assumes cache_path ends in .parquet
            base, ext = os.path.splitext(cache_path)
            cache_path = f"{base}_sample_{sample_size}{ext}"

        # 3. Extract or Load geometry features
        # This function handles the caching logic internally:
        # - Checks if cache_path exists and load_cached_data is True.
        # - If so, loads and merges.
        # - If not, computes features from .xyz files, saves to cache_path, and merges.
        df_merged = extract_geometry_features(
            metadata_df=df,
            cache_file_path=cache_path,
            load_cached_data=load_cached_data,
        )

        return df_merged

    def load_train_data(self, sample_size=None, load_cached_data=True):
        """
        Loads the training dataset.

        Args:
            sample_size (int, optional): Number of samples to load.
            load_cached_data (bool): Whether to use cached geometry features.

        Returns:
            pd.DataFrame: Training data with features.
        """
        return self._load_dataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            cache_path=Config.TRAIN_GEO_CACHE,
            sample_size=sample_size,
            load_cached_data=load_cached_data,
        )

    def load_val_data(self, sample_size=None, load_cached_data=True):
        """
        Loads the validation dataset.

        Args:
            sample_size (int, optional): Number of samples to load.
            load_cached_data (bool): Whether to use cached geometry features.

        Returns:
            pd.DataFrame: Validation data with features.
        """
        return self._load_dataset(
            metadata_path=Config.VAL_METADATA_PATH,
            cache_path=Config.VAL_GEO_CACHE,
            sample_size=sample_size,
            load_cached_data=load_cached_data,
        )

    def load_test_data(self, sample_size=None, load_cached_data=True):
        """
        Loads the test dataset.

        Args:
            sample_size (int, optional): Number of samples to load.
            load_cached_data (bool): Whether to use cached geometry features.

        Returns:
            pd.DataFrame: Test data with features.
        """
        return self._load_dataset(
            metadata_path=Config.TEST_METADATA_PATH,
            cache_path=Config.TEST_GEO_CACHE,
            sample_size=sample_size,
            load_cached_data=load_cached_data,
        )
