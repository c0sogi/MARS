import os
import gc
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SUBSAMPLE_SIZE,
    GRID_PRECISION,
    RANDOM_STATE,
    FARE_MIN,
    FARE_MAX,
)
from library.encoders import GlobalRouteEncoder
from library.features import FeatureEngineer
from library.utils import clamp_coordinates


class TaxiDataLoader:
    """
    Manages data loading, preprocessing, and caching for the taxi fare prediction task.
    Implements the Two-Stage Global-Local strategy:
    1. Global Feature Extraction (OOF) on the full dataset.
    2. Subsampling and Local Feature Engineering for stable training.
    """

    def __init__(self):
        self.train_path = TRAIN_PATH
        self.val_path = VAL_PATH
        self.test_path = TEST_PATH
        self.cache_dir = CACHE_DIR
        self.subsample_size = SUBSAMPLE_SIZE
        self.grid_precision = GRID_PRECISION
        self.random_state = RANDOM_STATE

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_raw_data(self):
        """Loads raw datasets from Parquet files."""
        print("Loading raw datasets...")
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)
        test_df = pd.read_parquet(self.test_path)
        return train_df, val_df, test_df

    def subsample_data(self, df, n_samples):
        """Subsamples the dataframe to a fixed size for stability."""
        if n_samples is not None and len(df) > n_samples:
            print(f"Subsampling training data from {len(df)} to {n_samples} rows...")
            return df.sample(n=n_samples, random_state=self.random_state).copy()
        return df

    def get_processed_data(self, load_cached_data=True):
        """
        Main pipeline execution method.
        Checks cache, otherwise processes data from scratch.

        Returns:
            X_train, y_train, X_val, y_val, X_test, test_keys
        """
        # Define cache file paths
        cache_files = {
            "X_train": os.path.join(self.cache_dir, "X_train.parquet"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.parquet"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.parquet"),
            "test_keys": os.path.join(self.cache_dir, "test_keys.npy"),
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
            print("Loading processed data from cache...")
            X_train = pd.read_parquet(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_val = pd.read_parquet(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            X_test = pd.read_parquet(cache_files["X_test"])
            test_keys = np.load(cache_files["test_keys"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_keys

        # 2. Process from Scratch
        print("Cache miss or force reload. Processing data from scratch...")

        # Load raw data
        train_df, val_df, test_df = self.load_raw_data()

        # Sanitize Target Variable
        # Cite solution_lesson_node_00017: Sanitize target to prevent L2 loss instability
        # Cite solution_lesson_node_00018: Sanitize validation to match training constraints and avoid outlier skew
        print(
            f"Sanitizing target variable (keeping {FARE_MIN} <= fare <= {FARE_MAX})..."
        )
        train_df = train_df[
            (train_df["fare_amount"] >= FARE_MIN)
            & (train_df["fare_amount"] <= FARE_MAX)
        ]
        val_df = val_df[
            (val_df["fare_amount"] >= FARE_MIN) & (val_df["fare_amount"] <= FARE_MAX)
        ]

        # Extract keys for submission
        test_keys = test_df["key"].values

        # --- STAGE 1: Global Feature Extraction (OOF) ---
        # We use the FULL (sanitized) training set here
        print("Stage 1: Applying Global Route Encoding (OOF)...")
        encoder = GlobalRouteEncoder(
            grid_precision=self.grid_precision,
            n_splits=5,
            random_state=self.random_state,
        )

        # Fit and transform full training set (OOF)
        train_df = encoder.fit_transform_oof(train_df)

        # Transform validation and test sets using the global map
        val_df = encoder.transform_global(val_df)
        test_df = encoder.transform_global(test_df)

        # --- STAGE 2: Subsampling & Feature Engineering ---
        # Subsample training data now that global stats are attached
        train_df = self.subsample_data(train_df, self.subsample_size)

        # Clean up memory
        gc.collect()

        print("Stage 2: Applying Feature Engineering...")
        # Initialize Feature Engineer with clamping enabled
        fe = FeatureEngineer(rotation_angle=45, clamp_input=True)

        train_df = fe.transform(train_df)
        val_df = fe.transform(val_df)
        test_df = fe.transform(test_df)

        # Define Feature Columns
        # We exclude keys, timestamps, and target
        exclude_cols = {
            "key",
            "pickup_datetime",
            "fare_amount",
            "p_lat_r",
            "p_lon_r",
            "d_lat_r",
            "d_lon_r",
        }
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Ensure consistent column order
        feature_cols = sorted(feature_cols)
        print(f"Final Features ({len(feature_cols)}): {feature_cols}")

        # Construct Final Arrays
        X_train = train_df[feature_cols]
        y_train = train_df["fare_amount"].values

        X_val = val_df[feature_cols]
        y_val = val_df["fare_amount"].values

        X_test = test_df[feature_cols]

        # 3. Save to Cache
        print("Saving processed data to cache...")
        X_train.to_parquet(cache_files["X_train"])
        np.save(cache_files["y_train"], y_train)
        X_val.to_parquet(cache_files["X_val"])
        np.save(cache_files["y_val"], y_val)
        X_test.to_parquet(cache_files["X_test"])
        np.save(cache_files["test_keys"], test_keys)

        return X_train, y_train, X_val, y_val, X_test, test_keys
