import os
import pandas as pd
import numpy as np
from library import config, utils


class TaxiDataProcessor:
    def __init__(self):
        """
        Initialize the Data Processor.
        Ensures the cache directory exists.
        """
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def clean_data(self, df, is_train=True):
        """
        Cleans the dataset by filtering out invalid rows.

        Args:
            df (pd.DataFrame): The input dataframe.
            is_train (bool): Flag indicating if this is training/validation data.
                             If True, rows with invalid coordinates or fares are dropped.
                             If False (test set), no rows are dropped to ensure submission integrity.

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        df = df.copy()

        # Only filter rows for training/validation sets
        # We must preserve all rows in the test set for submission
        if is_train:
            # 1. Coordinate Bounding Box Filter
            # Keep rows where pickup and dropoff are within NYC limits
            mask_coords = (
                (df["pickup_longitude"] >= config.NYC_MIN_LON)
                & (df["pickup_longitude"] <= config.NYC_MAX_LON)
                & (df["pickup_latitude"] >= config.NYC_MIN_LAT)
                & (df["pickup_latitude"] <= config.NYC_MAX_LAT)
                & (df["dropoff_longitude"] >= config.NYC_MIN_LON)
                & (df["dropoff_longitude"] <= config.NYC_MAX_LON)
                & (df["dropoff_latitude"] >= config.NYC_MIN_LAT)
                & (df["dropoff_latitude"] <= config.NYC_MAX_LAT)
            )
            df = df[mask_coords]

            # 2. Passenger Count Filter
            mask_pass = (df["passenger_count"] >= config.MIN_PASSENGER_COUNT) & (
                df["passenger_count"] <= config.MAX_PASSENGER_COUNT
            )
            df = df[mask_pass]

            # 3. Fare Amount Filter
            if "fare_amount" in df.columns:
                mask_fare = df["fare_amount"] > config.MIN_FARE_AMOUNT
                df = df[mask_fare]

        return df

    def engineer_features(self, df):
        """
        Generates new features from existing columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added features.
        """
        df = df.copy()

        # 1. Temporal Features
        # Ensure pickup_datetime is a datetime object
        if "pickup_datetime" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
                df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

            df["year"] = df["pickup_datetime"].dt.year
            df["month"] = df["pickup_datetime"].dt.month
            df["day"] = df["pickup_datetime"].dt.day
            df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
            df["hour"] = df["pickup_datetime"].dt.hour

        # 2. Spatial Features
        # Haversine Distance (Great Circle Distance)
        df["haversine_dist"] = utils.haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Coordinate Differences (Proxy for Manhattan Distance)
        df["abs_diff_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
        df["abs_diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

        # 3. Landmark Distances
        for name, (lat, lon) in config.NYC_LANDMARKS.items():
            df[f"dist_pickup_{name}"] = utils.haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"], lat, lon
            )
            df[f"dist_dropoff_{name}"] = utils.haversine_distance(
                df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
            )

        # 4. Rotated Coordinates (approx 29 degrees for NYC grid)
        # Rotating coordinates helps tree models split more effectively on the diagonal street grid
        rot_angle = np.radians(29)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)

        # Pickup
        df["pickup_rot_x"] = (
            df["pickup_longitude"] * cos_a - df["pickup_latitude"] * sin_a
        )
        df["pickup_rot_y"] = (
            df["pickup_longitude"] * sin_a + df["pickup_latitude"] * cos_a
        )

        # Dropoff
        df["dropoff_rot_x"] = (
            df["dropoff_longitude"] * cos_a - df["dropoff_latitude"] * sin_a
        )
        df["dropoff_rot_y"] = (
            df["dropoff_longitude"] * sin_a + df["dropoff_latitude"] * cos_a
        )

        return df

    def process_data(self, load_cached_data=True, train_sample_size=None):
        """
        Orchestrates the loading, cleaning, and feature engineering of Train, Validation, and Test sets.
        Implements caching to avoid re-processing large datasets.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from cache.
            train_sample_size (int, optional): If provided, samples the training set to this size.

        Returns:
            tuple: (train_df, val_df, test_df)
        """

        def _get_dataset(name, raw_path, cache_filename, is_train):
            cache_path = os.path.join(self.cache_dir, cache_filename)

            # Attempt to load from cache
            if load_cached_data and os.path.exists(cache_path):
                print(f"Loading cached {name} data from {cache_path}")
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    print(f"Error loading cache for {name}: {e}. Re-processing.")

            # Process from scratch
            print(f"Processing {name} data from {raw_path}...")
            df = pd.read_parquet(raw_path)

            # Clean (only if training/validation)
            df = self.clean_data(df, is_train=is_train)

            # Engineer features
            df = self.engineer_features(df)

            # Save to cache
            print(f"Saving processed {name} data to {cache_path}")
            df.to_parquet(cache_path, index=False)

            return df

        # Process datasets
        train_df = _get_dataset(
            "train", config.TRAIN_DATA_PATH, "train_processed.parquet", is_train=True
        )

        val_df = _get_dataset(
            "val", config.VAL_DATA_PATH, "val_processed.parquet", is_train=True
        )

        test_df = _get_dataset(
            "test", config.TEST_DATA_PATH, "test_processed.parquet", is_train=False
        )

        # Apply sampling to training data if requested (for debugging/fast iteration)
        if train_sample_size is not None:
            if len(train_df) > train_sample_size:
                print(
                    f"Sampling training data from {len(train_df)} to {train_sample_size} rows."
                )
                train_df = train_df.sample(
                    n=train_sample_size, random_state=config.SEED
                )

        return train_df, val_df, test_df
