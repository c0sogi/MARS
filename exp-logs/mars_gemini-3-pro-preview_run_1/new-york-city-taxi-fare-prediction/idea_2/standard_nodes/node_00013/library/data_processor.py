import os
import pandas as pd
import numpy as np
from library import config
from library import utils


class TaxiDataManager:
    """
    Manages data ingestion, cleaning, and deterministic feature engineering
    for the Taxi Fare Prediction task.
    """

    def __init__(self, cache_dir=config.CACHE_DIR):
        """
        Initialize the data manager.

        Args:
            cache_dir (str): Directory to store processed parquet files.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_and_clean(self, split_name, sample_frac=None):
        """
        Loads the raw data from metadata paths and performs basic cleaning.

        Args:
            split_name (str): One of 'train', 'val', 'test'.
            sample_frac (float, optional): Fraction of data to load for debugging.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        # Determine file path
        if split_name == "train":
            path = config.TRAIN_DATA_PATH
        elif split_name == "val":
            path = config.VAL_DATA_PATH
        elif split_name == "test":
            path = config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # Load data
        # Using pyarrow engine for speed
        df = pd.read_parquet(path)

        # Sampling for debugging/fast iteration
        if sample_frac is not None and 0 < sample_frac < 1.0:
            df = df.sample(
                frac=sample_frac, random_state=config.RANDOM_SEED
            ).reset_index(drop=True)

        # Cleaning Logic
        initial_len = len(df)

        # 1. Coordinate Validity
        # Latitudes must be between -90 and 90
        valid_lat = (df["pickup_latitude"].between(-90, 90)) & (
            df["dropoff_latitude"].between(-90, 90)
        )

        # Longitudes must be between -180 and 180
        valid_lon = (df["pickup_longitude"].between(-180, 180)) & (
            df["dropoff_longitude"].between(-180, 180)
        )

        df = df[valid_lat & valid_lon]

        # 2. Target Validity (only for train/val)
        if "fare_amount" in df.columns:
            # Fare must be positive
            df = df[df["fare_amount"] > 0]
            # Cite solution_lesson_node_00012: Filter extreme outliers to stabilize RMSE
            df = df[df["fare_amount"] < 500]

        # 3. Drop NaNs in critical columns
        cols_to_check = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
        df = df.dropna(subset=cols_to_check)

        return df

    def add_temporal_features(self, df):
        """
        Adds temporal features derived from pickup_datetime.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added temporal features.
        """
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        # Extract components
        df["year"] = df["pickup_datetime"].dt.year
        df["month"] = df["pickup_datetime"].dt.month
        df["day"] = df["pickup_datetime"].dt.day
        df["hour"] = df["pickup_datetime"].dt.hour
        df["weekday"] = df["pickup_datetime"].dt.dayofweek

        # Continuous time feature (Unix timestamp)
        # Useful for capturing inflation or long-term trends
        df["pickup_timestamp"] = df["pickup_datetime"].astype("int64") // 10**9

        return df

    def add_geometric_features(self, df):
        """
        Adds geometric features including distances, rotations, and landmark proximities.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added geometric features.
        """
        # 1. Basic Distances
        df["dist_haversine"] = utils.haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        df["dist_manhattan"] = utils.manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # 2. Coordinate Rotations (Grid Alignment)
        # Sum and difference of coordinates help tree models learn diagonal boundaries
        # which correspond to the rotated street grid of Manhattan.
        df["pickup_lat_plus_lon"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["pickup_lat_minus_lon"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_lat_plus_lon"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        df["dropoff_lat_minus_lon"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        # 3. Landmark Distances
        # Calculate distance from pickup and dropoff to specific points of interest
        for name, (lat, lon) in config.LANDMARKS.items():
            df[f"dist_pickup_{name}"] = utils.haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"], lat, lon
            )
            df[f"dist_dropoff_{name}"] = utils.haversine_distance(
                df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
            )

        return df

    def get_processed_data(self, split_name, load_cached_data=True, sample_frac=None):
        """
        Orchestrates the data loading, processing, and caching pipeline.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from cache first.
            sample_frac (float, optional): Fraction of data to use (passed to load_and_clean).

        Returns:
            pd.DataFrame: Processed dataframe ready for modeling.
        """
        cache_filename = f"processed_{split_name}.parquet"
        if sample_frac is not None:
            cache_filename = f"processed_{split_name}_sample_{sample_frac}.parquet"

        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)

                # Cite debug_lesson_1: Validate Cached Artifacts Against Current Configuration Schema
                # Create a minimal dummy dataframe to determine expected feature columns
                dummy_data = {
                    "pickup_datetime": ["2020-01-01 00:00:00 UTC"],
                    "pickup_longitude": [-74.0],
                    "pickup_latitude": [40.7],
                    "dropoff_longitude": [-73.9],
                    "dropoff_latitude": [40.8],
                    "passenger_count": [1],
                }
                dummy_df = pd.DataFrame(dummy_data)
                dummy_df = self.add_temporal_features(dummy_df)
                dummy_df = self.add_geometric_features(dummy_df)

                # Check if cached dataframe has all columns produced by current feature engineering
                missing_cols = [c for c in dummy_df.columns if c not in df.columns]
                if missing_cols:
                    print(
                        f"Cached data {cache_path} missing columns: {missing_cols}. Invalidating cache."
                    )
                    raise ValueError("Cache schema mismatch")

                return df
            except Exception as e:
                print(f"Could not load cache or validation failed: {e}")
                # If load fails, proceed to re-compute
                pass

        # 2. Compute from scratch
        # Load and Clean
        df = self.load_and_clean(split_name, sample_frac=sample_frac)

        # Feature Engineering
        df = self.add_temporal_features(df)
        df = self.add_geometric_features(df)

        # Memory Optimization
        df = utils.reduce_memory_usage(df)

        # 3. Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            # If save fails (e.g. disk full), just continue
            pass

        return df
