import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import haversine_distance


class DataFactory:
    """
    Handles loading, preprocessing, and caching of datasets.
    Implements physics-consistent filtering and subsampling strategies.
    """

    @staticmethod
    def apply_physics_filter(df):
        """
        Applies a strict physics-consistent filter to remove 'garbage' outliers.
        Removes rows where Fare > Threshold AND Fare/Km > Rate_Threshold.

        Args:
            df (pd.DataFrame): Input dataframe containing coordinates and fare_amount.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        # Ensure necessary columns exist
        required_cols = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "fare_amount",
        ]
        if not all(col in df.columns for col in required_cols):
            # If columns missing (e.g. test set has no fare), return as is or handle appropriately.
            # However, this filter is intended for data with targets.
            return df

        # Calculate Haversine distance in Kilometers
        dist_km = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Avoid division by zero by adding a small epsilon
        dist_km = np.maximum(dist_km, 1e-6)

        # Calculate Fare per Km
        fare_per_km = df["fare_amount"] / dist_km

        # Define Outlier Condition: High Fare AND High Rate
        # We only remove if BOTH conditions are met, preserving valid long trips.
        is_garbage = (df["fare_amount"] > Config.OUTLIER_FARE_THRESHOLD) & (
            fare_per_km > Config.OUTLIER_RATE_THRESHOLD
        )

        # Filter
        df_clean = df[~is_garbage].copy()

        return df_clean

    @classmethod
    def load_clean_full_train_data(cls):
        """
        Loads the full training dataset and applies the physics filter.
        Used for generating the Global Knowledge Base (Stage 1).
        Does not cache to disk to avoid duplicating the massive dataset.
        """
        print(f"Loading full training data from {Config.TRAIN_DATA_PATH}...")
        df = pd.read_parquet(Config.TRAIN_DATA_PATH)

        print("Applying physics-consistent filtering...")
        df = cls.apply_physics_filter(df)

        print(f"Full clean training set size: {len(df)}")
        return df

    @classmethod
    def load_train_data(cls, load_cached_data=True):
        """
        Loads the training data for the learner (Stage 2).
        1. Checks cache.
        2. If not cached: Loads full data, filters, subsamples, and caches.
        """
        cache_path = Config.PROCESSED_TRAIN_CACHE_PATH

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached training data from {cache_path}...")
            return pd.read_parquet(cache_path)

        # 2. Process from scratch
        print("Cache not found or ignored. Processing training data...")

        # Use the helper to get clean full data
        df = cls.load_clean_full_train_data()

        # Subsample
        target_size = Config.TRAIN_SUBSAMPLE_SIZE
        if len(df) > target_size:
            print(f"Subsampling training data to {target_size} rows...")
            df = df.sample(n=target_size, random_state=Config.RANDOM_SEED)
        else:
            print(
                f"Dataset smaller than subsample target ({len(df)} < {target_size}). Using all data."
            )

        # Save to cache
        print(f"Saving processed training data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

        return df

    @classmethod
    def load_val_data(cls, load_cached_data=True):
        """
        Loads the validation data.
        1. Checks cache.
        2. If not cached: Loads full val data, filters, and caches.
        """
        cache_path = Config.PROCESSED_VAL_CACHE_PATH

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached validation data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Cache not found or ignored. Processing validation data...")
        df = pd.read_parquet(Config.VAL_DATA_PATH)

        # Apply filtering to validation as well to ensure metrics reflect performance on valid rides
        print("Applying physics-consistent filtering to validation set...")
        df = cls.apply_physics_filter(df)

        print(f"Saving processed validation data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

        return df

    @classmethod
    def load_test_data(cls, load_cached_data=True):
        """
        Loads the test data.
        1. Checks cache.
        2. If not cached: Loads raw test data and caches.
        Note: Test data does not have targets, so no physics filtering is applied.
        """
        cache_path = Config.PROCESSED_TEST_CACHE_PATH

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached test data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Cache not found or ignored. Processing test data...")
        df = pd.read_parquet(Config.TEST_DATA_PATH)

        print(f"Saving processed test data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

        return df
