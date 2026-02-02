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

    def apply_strict_filter(self, df):
        """
        Applies Strict Hygiene for the Background Knowledge Base.
        Removes outliers and physically inconsistent rides to create clean priors.
        """
        # 1. Fare Amount Bounds
        mask_fare = (df["fare_amount"] >= self.config.BG_MIN_FARE) & (
            df["fare_amount"] <= self.config.BG_MAX_FARE
        )
        df = df[mask_fare]

        # 2. Fare Per Km Check (Physics Consistency)
        # Calculate distance (km)
        dists = self._calculate_trip_distance(df)

        # Filter: Fare <= Max_Rate * Distance
        # We handle the edge case of distance=0 implicitly:
        # If dist=0, max_fare allowed is 0. Since min_fare is 2.50, these rows are removed.
        # This correctly filters static GPS noise with high fares.
        mask_rate = df["fare_amount"] <= (self.config.BG_MAX_FARE_PER_KM * dists)

        df = df[mask_rate]

        # Cleanup
        del dists, mask_fare, mask_rate
        clean_memory()

        return df

    def apply_loose_filter(self, df):
        """
        Applies Loose Hygiene for the Foreground Training Set.
        Retains valid heavy-tail outliers to allow the model to learn robustly.
        """
        # Basic Fare Bounds only
        mask = (df["fare_amount"] >= self.config.FG_MIN_FARE) & (
            df["fare_amount"] <= self.config.FG_MAX_FARE
        )
        df = df[mask]

        clean_memory()
        return df

    def split_background_foreground(self, df):
        """
        Partitions the dataset into disjoint Background and Foreground sets
        based on the sizes defined in Config.
        """
        total_rows = len(df)
        bg_target = self.config.BACKGROUND_SIZE
        fg_target = self.config.FOREGROUND_SIZE

        # Adjust sizes if dataset is smaller than config (e.g. during debugging)
        if bg_target + fg_target > total_rows:
            scale = total_rows / (bg_target + fg_target)
            bg_target = int(bg_target * scale)
            fg_target = int(fg_target * scale)

        # Deterministic Shuffle
        indices = np.arange(total_rows)
        rng = np.random.RandomState(self.config.SEED)
        rng.shuffle(indices)

        # Slice indices
        bg_indices = indices[:bg_target]
        fg_indices = indices[bg_target : bg_target + fg_target]

        # Create DataFrames
        background_df = df.iloc[bg_indices].copy()
        foreground_df = df.iloc[fg_indices].copy()

        del indices, bg_indices, fg_indices
        clean_memory()

        return background_df, foreground_df

    def get_data(self, load_cached_data=True):
        """
        Main entry point. Loads data, processes it (split + filter), and returns DataFrames.
        Implements caching to disk.

        Returns:
            background_df, foreground_df, val_df, test_df
        """
        # Define cache file paths
        cache_bg = self.config.get_cache_path("background.parquet")
        cache_fg = self.config.get_cache_path("foreground.parquet")
        cache_val = self.config.get_cache_path("val_processed.parquet")
        cache_test = self.config.get_cache_path("test_processed.parquet")

        # 1. Try Loading Cache
        if (
            load_cached_data
            and os.path.exists(cache_bg)
            and os.path.exists(cache_fg)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):

            print("Loading cached data from working directory...")
            try:
                background_df = pd.read_parquet(cache_bg)
                foreground_df = pd.read_parquet(cache_fg)
                val_df = pd.read_parquet(cache_val)
                test_df = pd.read_parquet(cache_test)
                return background_df, foreground_df, val_df, test_df
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

        # Split
        print("Splitting training data into Background and Foreground...")
        background_df, foreground_df = self.split_background_foreground(train_df)

        # Free memory of raw train
        del train_df
        clean_memory()

        # Apply Hygiene
        print("Applying Strict Filter to Background Knowledge Base...")
        print(f"  Rows before: {len(background_df)}")
        background_df = self.apply_strict_filter(background_df)
        print(f"  Rows after:  {len(background_df)}")

        print("Applying Loose Filter to Foreground Training Set...")
        print(f"  Rows before: {len(foreground_df)}")
        foreground_df = self.apply_loose_filter(foreground_df)
        print(f"  Rows after:  {len(foreground_df)}")

        # Process Validation
        # Apply Loose Filter to validation to remove impossible values (e.g. negative fares)
        # while keeping it representative of the test distribution.
        val_df = self.apply_loose_filter(val_df)

        # Subsample Validation if configured (for speed)
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
        self.config.setup_dirs()  # Ensure directory exists
        background_df.to_parquet(cache_bg, index=False)
        foreground_df.to_parquet(cache_fg, index=False)
        val_df.to_parquet(cache_val, index=False)
        test_df.to_parquet(cache_test, index=False)

        return background_df, foreground_df, val_df, test_df
