import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clean_memory, haversine_distance


class DataLoader:
    """
    Handles data ingestion, partitioning, hygiene, and caching for the
    Disjoint Background-Foreground strategy.
    """

    def __init__(self, config: Config):
        self.config = config

    def _calculate_trip_distance(self, df):
        """
        Helper to calculate Haversine distance for filtering logic.
        Returns distance in kilometers.
        """
        return haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

    def apply_filter(self, df, strict=True):
        """
        Applies hygiene filters to the dataset.
        """
        # 1. Fare Amount Bounds
        mask_fare = (df["fare_amount"] >= self.config.MIN_FARE) & (
            df["fare_amount"] <= self.config.MAX_FARE
        )
        df = df[mask_fare]

        if strict:
            # 2. Fare Per Km Check (Physics Consistency)
            # Only apply this if we want to be very strict about data quality for stats
            dists = self._calculate_trip_distance(df)
            mask_rate = df["fare_amount"] <= (self.config.MAX_FARE_PER_KM * dists)
            df = df[mask_rate]
            del dists, mask_rate

        # Cleanup
        del mask_fare
        clean_memory()

        return df

    def get_data(self, load_cached_data=True):
        """
        Main entry point. Loads data, processes it, and returns DataFrames.
        Implements caching to disk.

        Returns:
            train_df, val_df, test_df
        """
        # Define cache file paths
        cache_train = self.config.get_cache_path("train_filtered.parquet")
        cache_val = self.config.get_cache_path("val_filtered.parquet")
        cache_test = self.config.get_cache_path("test_filtered.parquet")

        # 1. Try Loading Cache
        if (
            load_cached_data
            and os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            print("Loading cached data from working directory...")
            try:
                train_df = pd.read_parquet(cache_train)
                val_df = pd.read_parquet(cache_val)
                test_df = pd.read_parquet(cache_test)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data.")

        # 2. Process from Scratch
        print("Processing data from scratch...")

        # Load Raw Data
        print(f"Loading raw train from {self.config.TRAIN_DATA_PATH}...")
        train_df = pd.read_parquet(self.config.TRAIN_DATA_PATH)

        print(f"Loading raw val from {self.config.VAL_DATA_PATH}...")
        val_df = pd.read_parquet(self.config.VAL_DATA_PATH)

        print(f"Loading raw test from {self.config.TEST_DATA_PATH}...")
        test_df = pd.read_parquet(self.config.TEST_DATA_PATH)

        # Apply Hygiene
        print("Applying Filter to Training Set...")
        print(f"  Rows before: {len(train_df)}")
        train_df = self.apply_filter(train_df, strict=True)
        print(f"  Rows after:  {len(train_df)}")

        # Process Validation
        # Apply strict filter to validation as well to ensure metric reflects
        # performance on valid rides, or loose if we expect dirty test data.
        # Given the task is RMSE, outliers hurt. We filter Val to match Train distribution
        # for reliable hyperparam tuning.
        val_df = self.apply_filter(val_df, strict=False)

        # Subsample Validation if configured
        if (
            self.config.VAL_SUBSET_SIZE is not None
            and len(val_df) > self.config.VAL_SUBSET_SIZE
        ):
            print(
                f"Subsampling validation set to {self.config.VAL_SUBSET_SIZE} rows..."
            )
            val_df = val_df.sample(
                n=self.config.VAL_SUBSET_SIZE, random_state=self.config.SEED
            )

        # Save to Cache
        print("Saving processed data to cache...")
        self.config.setup_dirs()
        train_df.to_parquet(cache_train, index=False)
        val_df.to_parquet(cache_val, index=False)
        test_df.to_parquet(cache_test, index=False)

        return train_df, val_df, test_df
