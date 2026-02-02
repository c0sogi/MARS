import os
import pandas as pd
from library.config import Config
from library.feature_engineering import generate_features


class DataManager:
    """
    Manages data loading, subsetting (debugging), and feature generation orchestration.
    Interacts with the feature_engineering library to produce cached parquet files.
    """

    @staticmethod
    def _get_metadata_path_and_cache_name(original_meta_path, split_name, size=None):
        """
        Determines the metadata file to use and the output cache filename.
        If size is specified and smaller than the full dataset, creates a temporary
        subset metadata file to support debugging/fast prototyping.
        """
        # Default behavior: Use full dataset
        if size is None:
            return original_meta_path, f"{split_name}_features.parquet"

        # Load original metadata to check size
        if not os.path.exists(original_meta_path):
            raise FileNotFoundError(f"Metadata file not found: {original_meta_path}")

        full_df = pd.read_csv(original_meta_path)

        # If requested size is larger or equal to full data, treat as normal
        if size >= len(full_df):
            return original_meta_path, f"{split_name}_features.parquet"

        # Create a deterministic subset
        subset_df = full_df.sample(n=size, random_state=Config.SEED).reset_index(
            drop=True
        )

        # Save temporary metadata to the cache directory
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        temp_meta_name = f"{split_name}_meta_debug_{size}.csv"
        temp_meta_path = os.path.join(Config.CACHE_DIR, temp_meta_name)
        subset_df.to_csv(temp_meta_path, index=False)

        # Define a unique cache name for this subset configuration
        cache_name = f"{split_name}_features_debug_{size}.parquet"

        return temp_meta_path, cache_name

    @classmethod
    def get_train_data(cls, size=None, load_cached_data=True):
        """
        Generates or loads training data.

        Args:
            size (int, optional): Number of samples to load (for debugging).
            load_cached_data (bool): Whether to attempt loading from parquet cache.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable (time_to_eruption).
        """
        meta_path, cache_name = cls._get_metadata_path_and_cache_name(
            Config.TRAIN_META_PATH, "train", size
        )

        df = generate_features(meta_path, cache_name, load_cached_data=load_cached_data)

        if "time_to_eruption" not in df.columns:
            raise ValueError("Training data must contain 'time_to_eruption' column.")

        y = df["time_to_eruption"]
        # Drop metadata columns to leave only features
        X = df.drop(columns=["segment_id", "time_to_eruption"])

        return X, y

    @classmethod
    def get_val_data(cls, size=None, load_cached_data=True):
        """
        Generates or loads validation data.

        Args:
            size (int, optional): Number of samples to load.
            load_cached_data (bool): Whether to attempt loading from parquet cache.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable (time_to_eruption).
        """
        meta_path, cache_name = cls._get_metadata_path_and_cache_name(
            Config.VAL_META_PATH, "val", size
        )

        df = generate_features(meta_path, cache_name, load_cached_data=load_cached_data)

        if "time_to_eruption" not in df.columns:
            raise ValueError("Validation data must contain 'time_to_eruption' column.")

        y = df["time_to_eruption"]
        X = df.drop(columns=["segment_id", "time_to_eruption"])

        return X, y

    @classmethod
    def get_test_data(cls, size=None, load_cached_data=True):
        """
        Generates or loads test data.

        Args:
            size (int, optional): Number of samples to load.
            load_cached_data (bool): Whether to attempt loading from parquet cache.

        Returns:
            X (pd.DataFrame): Feature matrix.
            ids (pd.Series): Segment IDs corresponding to the features.
        """
        meta_path, cache_name = cls._get_metadata_path_and_cache_name(
            Config.TEST_META_PATH, "test", size
        )

        df = generate_features(meta_path, cache_name, load_cached_data=load_cached_data)

        ids = df["segment_id"]

        # Prepare feature matrix
        # Test data might not have time_to_eruption, but if it exists (e.g. local test), drop it
        cols_to_drop = ["segment_id"]
        if "time_to_eruption" in df.columns:
            cols_to_drop.append("time_to_eruption")

        X = df.drop(columns=cols_to_drop)

        return X, ids
