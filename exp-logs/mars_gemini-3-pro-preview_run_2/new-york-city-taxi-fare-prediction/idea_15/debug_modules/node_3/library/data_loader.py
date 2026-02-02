import os
import pandas as pd
import numpy as np
from library.config import Config
from library.geometry_utils import DistanceCalculator


class TaxiDataLoader:
    """
    Handles data ingestion, hygiene, and splitting for the Taxi Fare Prediction task.
    Implements the Dual-Hygiene strategy:
    - Wisdom Set: Full dataset with strict filtering for robust statistics.
    - Learner Set: Subsampled dataset with loose filtering to learn heavy tails.
    """

    def __init__(self):
        self.rng = np.random.default_rng(Config.SEED)
        # Define internal cache path for the deterministic Learner split
        self.learner_cache_path = os.path.join(
            Config.WORKING_DIR, "learner_split.parquet"
        )

    def _process_datetime(self, df):
        """Converts pickup_datetime to datetime objects."""
        if "pickup_datetime" in df.columns:
            # Coerce errors to handle potential garbage, though metadata is clean
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)
        return df

    def _calculate_metrics(self, df):
        """
        Calculates distance and fare_per_km for filtering.
        Adds 'dist_km' column to the dataframe temporarily.
        """
        # Calculate Haversine distance
        df["dist_km"] = DistanceCalculator.haversine(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )
        return df

    def apply_strict_filters(self, df):
        """
        Applies strict filtering for the Wisdom Set (Statistics Generation).
        Removes noise and outliers to ensure clean priors.
        """
        filters = Config.STRICT_FILTER

        # Ensure metrics exist
        if "dist_km" not in df.columns:
            df = self._calculate_metrics(df)

        # Calculate rate (handle division by zero by producing NaN/Inf which are filtered)
        # We use a small epsilon or just rely on the comparison
        with np.errstate(divide="ignore", invalid="ignore"):
            fare_per_km = df["fare_amount"] / df["dist_km"]

        # Create mask
        mask = (
            (df["fare_amount"] >= filters["min_fare"])
            & (df["fare_amount"] <= filters["max_fare"])
            & (df["dist_km"] >= filters["min_dist_km"])
            & (df["passenger_count"] >= filters["min_passenger"])
            & (fare_per_km <= filters["max_fare_per_km"])
        )

        return df[mask].copy()

    def apply_loose_filters(self, df):
        """
        Applies loose filtering for the Learner Set (Model Training).
        Retains valid high-fare trips (heavy tails).
        """
        filters = Config.LOOSE_FILTER

        if "dist_km" not in df.columns:
            df = self._calculate_metrics(df)

        with np.errstate(divide="ignore", invalid="ignore"):
            fare_per_km = df["fare_amount"] / df["dist_km"]

        # Handle 0 distance for rate check: if dist is 0, rate is Inf.
        # If max_fare_per_km is set, Inf will be filtered out.
        # However, LOOSE_FILTER allows min_dist_km=0.0.
        # If dist=0 and fare > 0, rate is Inf. We likely want to keep these if they are valid surcharges?
        # But Config says max_fare_per_km=500. So we filter Inf.

        mask = (
            (df["fare_amount"] >= filters["min_fare"])
            & (df["fare_amount"] <= filters["max_fare"])
            & (df["dist_km"] >= filters["min_dist_km"])
            & (df["passenger_count"] >= filters["min_passenger"])
        )

        # Only apply rate filter where distance is non-zero to avoid aggressive filtering of valid 0-dist trips?
        # Or strictly follow config. Config says max_fare_per_km=500.
        # We will assume strictly following config.
        # If dist=0, fare_per_km=Inf, so it is > 500, so it gets filtered.
        # This is consistent with removing bad data.

        # Check for NaN/Inf in rate
        valid_rate = (fare_per_km <= filters["max_fare_per_km"]) | (df["dist_km"] == 0)
        # Note: If dist=0, we bypass rate check if we want to respect min_dist_km=0.
        # But usually dist=0 is bad data. Let's stick to the logic that if rate > 500 it's bad.

        mask = mask & (fare_per_km <= filters["max_fare_per_km"])

        return df[mask].copy()

    def load_datasets(self, load_cached_data=True, debug=False):
        """
        Loads and processes the datasets.

        Args:
            load_cached_data (bool): If True, tries to load the cached Learner split.
            debug (bool): If True, uses small subsamples for all datasets.

        Returns:
            tuple: (learner_df, wisdom_df, val_df, test_df)
        """
        print("Loading datasets...")

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Load Test and Val (smaller files)
        test_df = pd.read_parquet(Config.TEST_DATA_PATH)
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)

        # Preprocess Test/Val
        test_df = self._process_datetime(test_df)
        val_df = self._process_datetime(val_df)
        val_df = self._calculate_metrics(
            val_df
        )  # Metrics needed for evaluation analysis if desired

        if debug:
            print(f"Debug mode: Subsampling Val and Test to {Config.DEBUG_SAMPLE_SIZE}")
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Prepare Learner Set (Training Data)
        learner_df = None

        # Try loading cache
        if load_cached_data and not debug and os.path.exists(self.learner_cache_path):
            print(f"Loading cached Learner set from {self.learner_cache_path}")
            learner_df = pd.read_parquet(self.learner_cache_path)
        else:
            print("Processing Learner set from raw data...")
            # Load full train to sample from
            train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)

            if debug:
                train_full = train_full.head(
                    Config.DEBUG_SAMPLE_SIZE * 5
                )  # Load a bit more to sample from
                sample_size = min(len(train_full), Config.DEBUG_SAMPLE_SIZE)
            else:
                sample_size = Config.LEARNER_SAMPLE_SIZE

            # Sample indices for Learner
            # We sample from the raw data first, then filter.
            # This ensures we are drawing from the original distribution.
            if len(train_full) > sample_size:
                learner_indices = self.rng.choice(
                    train_full.index, size=sample_size, replace=False
                )
                learner_raw = train_full.loc[learner_indices].copy()
            else:
                learner_raw = train_full.copy()

            # Preprocess and Filter Learner
            learner_raw = self._process_datetime(learner_raw)
            learner_df = self.apply_loose_filters(learner_raw)

            # Cache the result (only if not debugging)
            if not debug and load_cached_data:
                print(f"Caching Learner set to {self.learner_cache_path}")
                learner_df.to_parquet(self.learner_cache_path)

        # 3. Prepare Wisdom Set (Statistics Data)
        # We always load this fresh or from memory because it's the full dataset (minus what we might want to exclude, but here we use full)
        # To save memory, we can reload train_full if we dropped it, or reuse it.
        # In a 220GB RAM environment, we can keep it.

        print("Processing Wisdom set...")
        if "train_full" not in locals():
            train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)
            if debug:
                train_full = train_full.head(Config.DEBUG_SAMPLE_SIZE)

        # Preprocess Wisdom
        # We need datetime for the "Spatiotemporal Rate" (Hour/Weekday)
        train_full = self._process_datetime(train_full)

        # Apply Strict Filters
        wisdom_df = self.apply_strict_filters(train_full)

        print(f"Dataset Shapes:")
        print(f"  Learner: {learner_df.shape}")
        print(f"  Wisdom:  {wisdom_df.shape}")
        print(f"  Val:     {val_df.shape}")
        print(f"  Test:    {test_df.shape}")

        return learner_df, wisdom_df, val_df, test_df
