import os
import gc
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    SUBSAMPLE_SIZE,
    RANDOM_SEED,
)
from library.spatial_engine import SpatialEngine
from library.feature_engineering import FeatureEngineer


class DataManager:
    """
    Orchestrates the data pipeline for the NYC Taxi Fare Prediction task.
    Manages data loading, cleaning, spatial prior generation, and feature engineering.
    """

    def __init__(self):
        self.spatial_engine = SpatialEngine()
        self.feature_engineer = FeatureEngineer()
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_and_clean_data(self, path):
        """
        Loads dataset from parquet and performs basic hygiene.
        Filters out negative fares or zero fares if target exists.
        """
        print(f"Loading data from {path}...")
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

        # Basic hygiene for training/validation data
        if "fare_amount" in df.columns:
            initial_len = len(df)
            # Filter non-positive fares (garbage)
            # We keep high fares as per instructions, but remove impossible negative/zero fares
            df = df[df["fare_amount"] > 0].reset_index(drop=True)
            dropped = initial_len - len(df)
            if dropped > 0:
                print(f"Dropped {dropped} rows with non-positive fare_amount.")

        return df

    def prepare_training_data(self, load_cached_data=True):
        """
        Prepares the training dataset:
        1. Loads full data.
        2. Generates Multi-Resolution Priors (Vectorized Subtraction).
        3. Subsamples to stable size.
        4. Adds physics/datetime features.
        """
        final_cache_path = os.path.join(self.cache_dir, "train_final.parquet")

        if load_cached_data and os.path.exists(final_cache_path):
            print(f"Loading cached final training data from {final_cache_path}")
            return pd.read_parquet(final_cache_path)

        print("Preparing Training Data Pipeline...")

        # 1. Load Full Data
        # We need the full dataset to generate high-quality priors
        # Optimization: If spatial cache exists, SpatialEngine will load it,
        # but we load raw here to be safe and compliant with API.
        # 220GB RAM allows holding this in memory.
        df = self.load_and_clean_data(TRAIN_DATA_PATH)

        # 2. Generate Spatial Priors
        # This step handles its own caching of the intermediate full dataset with priors
        df = self.spatial_engine.generate_kfold_priors(
            df, load_cached_data=load_cached_data
        )

        # 3. Subsample
        # We subsample AFTER generating priors to ensure priors are based on full data density
        if len(df) > SUBSAMPLE_SIZE:
            print(
                f"Subsampling training data from {len(df)} to {SUBSAMPLE_SIZE} rows..."
            )
            df = df.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(
                drop=True
            )

        # 4. Feature Engineering
        df = self.feature_engineer.add_datetime_features(df)
        df = self.feature_engineer.add_physics_features(df)

        # 5. Save Cache
        print(f"Saving final training data to {final_cache_path}")
        df.to_parquet(final_cache_path, index=False)

        gc.collect()
        return df

    def prepare_validation_data(self, full_train_df=None, load_cached_data=True):
        """
        Prepares the validation dataset.
        Uses global stats from the full training set (passed or loaded).
        """
        final_cache_path = os.path.join(self.cache_dir, "val_final.parquet")

        if load_cached_data and os.path.exists(final_cache_path):
            print(f"Loading cached final validation data from {final_cache_path}")
            return pd.read_parquet(final_cache_path)

        print("Preparing Validation Data Pipeline...")

        # Load Validation Data
        val_df = self.load_and_clean_data(VAL_DATA_PATH)

        # Ensure we have full training data for statistics
        if full_train_df is None:
            print("Loading full training data for validation statistics...")
            full_train_df = self.load_and_clean_data(TRAIN_DATA_PATH)

        # Apply Global Priors
        # Note: Validation set is treated as unseen, so we use global stats from train
        val_df = self.spatial_engine.apply_global_priors(val_df, full_train_df)

        # Feature Engineering
        val_df = self.feature_engineer.add_datetime_features(val_df)
        val_df = self.feature_engineer.add_physics_features(val_df)

        # Save Cache
        print(f"Saving final validation data to {final_cache_path}")
        val_df.to_parquet(final_cache_path, index=False)

        gc.collect()
        return val_df

    def prepare_test_data(self, full_train_df=None, load_cached_data=True):
        """
        Prepares the test dataset.
        Uses global stats from the full training set.
        """
        final_cache_path = os.path.join(self.cache_dir, "test_final.parquet")

        if load_cached_data and os.path.exists(final_cache_path):
            print(f"Loading cached final test data from {final_cache_path}")
            return pd.read_parquet(final_cache_path)

        print("Preparing Test Data Pipeline...")

        # Load Test Data (No cleaning required for target as it doesn't exist)
        test_df = pd.read_parquet(TEST_DATA_PATH)

        # Ensure we have full training data for statistics
        if full_train_df is None:
            print("Loading full training data for test statistics...")
            full_train_df = self.load_and_clean_data(TRAIN_DATA_PATH)

        # Apply Global Priors
        test_df = self.spatial_engine.apply_global_priors(test_df, full_train_df)

        # Feature Engineering
        test_df = self.feature_engineer.add_datetime_features(test_df)
        test_df = self.feature_engineer.add_physics_features(test_df)

        # Save Cache
        print(f"Saving final test data to {final_cache_path}")
        test_df.to_parquet(final_cache_path, index=False)

        gc.collect()
        return test_df
