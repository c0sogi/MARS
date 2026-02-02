import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import haversine_array, rotate_coordinates


class DataProcessor:
    """
    Handles data ingestion, cleaning, and basic feature extraction for the Taxi Fare Prediction task.
    Implements strict sanitation and feature engineering as per Idea 4.
    """

    def __init__(self):
        # Ensure working directory exists for caching artifacts
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        # Set random seed for reproducibility in sampling
        np.random.seed(Config.SEED)

    def load_data(self, path):
        """
        Loads data from a Parquet file.
        """
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

    def clean_data(self, df, mode="train"):
        """
        Applies strict bounding box filtering and fare amount filtering.

        Args:
            df: DataFrame to clean.
            mode: 'train', 'val', or 'test'.
                  - 'train': Drop spatial outliers, filter fare amount.
                  - 'val': Drop spatial outliers to ensure metric stability.
                  - 'test': Clip spatial outliers to bounding box (preserve rows for submission).
        """
        # 1. Spatial Cleaning
        # Define bounds from Config
        lat_min, lat_max = Config.BB_LAT_MIN, Config.BB_LAT_MAX
        lon_min, lon_max = Config.BB_LON_MIN, Config.BB_LON_MAX

        if mode == "test":
            # Clip coordinates to bounding box to preserve all rows for submission
            # This prevents distance feature explosion while keeping the key valid
            df["pickup_longitude"] = df["pickup_longitude"].clip(lon_min, lon_max)
            df["pickup_latitude"] = df["pickup_latitude"].clip(lat_min, lat_max)
            df["dropoff_longitude"] = df["dropoff_longitude"].clip(lon_min, lon_max)
            df["dropoff_latitude"] = df["dropoff_latitude"].clip(lat_min, lat_max)
        else:
            # Drop rows outside the bounding box for Train and Val
            # This ensures the model trains and evaluates on valid NYC data
            mask = (
                (df["pickup_longitude"].between(lon_min, lon_max))
                & (df["pickup_latitude"].between(lat_min, lat_max))
                & (df["dropoff_longitude"].between(lon_min, lon_max))
                & (df["dropoff_latitude"].between(lat_min, lat_max))
            )
            df = df[mask].copy()

        # 2. Target Variable Cleaning (Train only)
        if mode == "train":
            # Filter fare amount outliers (e.g., negative fares or extreme highs)
            fare_min, fare_max = Config.FARE_MIN, Config.FARE_MAX
            mask_fare = df["fare_amount"].between(fare_min, fare_max)
            df = df[mask_fare].copy()

        return df

    def add_basic_features(self, df):
        """
        Generates temporal features, landmark distances, and rotated coordinates.
        """
        # Ensure pickup_datetime is a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        # 1. Temporal Features
        df["hour"] = df["pickup_datetime"].dt.hour
        df["year"] = df["pickup_datetime"].dt.year
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

        # 2. Coordinate Rotation
        # Align with NYC street grid using the utility function
        p_lon_rot, p_lat_rot = rotate_coordinates(
            df["pickup_longitude"].values, df["pickup_latitude"].values
        )
        d_lon_rot, d_lat_rot = rotate_coordinates(
            df["dropoff_longitude"].values, df["dropoff_latitude"].values
        )

        df["pickup_lon_rot"] = p_lon_rot
        df["pickup_lat_rot"] = p_lat_rot
        df["dropoff_lon_rot"] = d_lon_rot
        df["dropoff_lat_rot"] = d_lat_rot

        # 3. Landmark Distances
        # Calculate Haversine distance to specific landmarks defined in Config
        for name, (lat, lon) in Config.LANDMARKS.items():
            # Distance from Pickup to Landmark
            df[f"dist_pickup_{name}"] = haversine_array(
                df["pickup_latitude"].values, df["pickup_longitude"].values, lat, lon
            )
            # Distance from Dropoff to Landmark
            df[f"dist_dropoff_{name}"] = haversine_array(
                df["dropoff_latitude"].values, df["dropoff_longitude"].values, lat, lon
            )

        # 4. Trip Distance (Direct Haversine)
        df["haversine_dist"] = haversine_array(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        return df

    def process_data(self, load_cached_data=True, sample_size=None):
        """
        Orchestrates the loading, cleaning, and feature engineering pipeline.
        Implements caching mechanism.

        Args:
            load_cached_data (bool): If True, attempts to load processed files from disk.
            sample_size (int, optional): If provided, limits the dataset size for debugging.

        Returns:
            train_df, val_df, test_df
        """
        # Define cache paths
        train_cache = Config.TRAIN_PROCESSED_PATH
        val_cache = Config.VAL_PROCESSED_PATH
        test_cache = Config.TEST_PROCESSED_PATH

        # Check if cache exists
        cache_exists = (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        )

        if load_cached_data and cache_exists:
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            # Apply sampling if requested (even on cached data for quick debugging)
            if sample_size is not None:
                print(f"Sampling {sample_size} rows from cached data...")
                train_df = train_df.head(sample_size)
                val_df = val_df.head(sample_size)

            return train_df, val_df, test_df

        print("Processing data from scratch...")

        # Load Raw Data
        print("Loading raw data...")
        train_df = self.load_data(Config.TRAIN_PATH)
        val_df = self.load_data(Config.VAL_PATH)
        test_df = self.load_data(Config.TEST_PATH)

        # Apply sampling if requested (before expensive processing)
        if sample_size is not None:
            print(f"Sampling {sample_size} rows...")
            train_df = train_df.head(sample_size)
            val_df = val_df.head(sample_size)
            # We typically keep the full test set for submission integrity,
            # but for pure pipeline debugging, one might sample it.
            # Here we keep test intact unless strictly debugging.

        # Clean Data
        print("Cleaning data...")
        train_df = self.clean_data(train_df, mode="train")
        val_df = self.clean_data(val_df, mode="val")
        test_df = self.clean_data(test_df, mode="test")

        # Add Features
        print("Adding basic features...")
        train_df = self.add_basic_features(train_df)
        val_df = self.add_basic_features(val_df)
        test_df = self.add_basic_features(test_df)

        # Save to Cache (only if not a debug sample, to avoid overwriting full cache with partial data)
        if sample_size is None:
            print("Saving processed data to cache...")
            train_df.to_parquet(train_cache, index=False)
            val_df.to_parquet(val_cache, index=False)
            test_df.to_parquet(test_cache, index=False)
        else:
            print("Skipping cache save due to active sampling.")

        return train_df, val_df, test_df
