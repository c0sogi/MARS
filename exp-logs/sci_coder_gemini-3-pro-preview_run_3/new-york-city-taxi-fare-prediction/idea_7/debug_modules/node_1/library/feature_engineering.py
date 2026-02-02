import os
import pandas as pd
import numpy as np
from library.config import (
    WORKING_DIR,
    ROTATION_ANGLE_RAD,
    LANDMARKS,
    AIRPORT_BOXES,
)
from library.utils import (
    haversine_distance,
    rotate_coordinates,
    manhattan_distance,
    reduce_mem_usage,
)


class FeatureEngineer:
    """
    Handles feature engineering for the NYC Taxi Fare Prediction task.
    Implements caching to speed up iterative development.
    """

    def __init__(self, load_cached_data=True):
        """
        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.
        """
        self.load_cached_data = load_cached_data
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

    def add_time_features(self, df):
        """
        Extracts temporal features from pickup_datetime.
        """
        # Ensure datetime type
        if df["pickup_datetime"].dtype == "object":
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        # Extract raw integer features (Strategic Retention)
        df["hour"] = df["pickup_datetime"].dt.hour
        df["day"] = df["pickup_datetime"].dt.day
        df["month"] = df["pickup_datetime"].dt.month
        df["year"] = df["pickup_datetime"].dt.year
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

        return df

    def add_rotated_features(self, df):
        """
        Adds rotated coordinates to align with NYC street grid.
        """
        # Rotate Pickup
        lat_rot, lon_rot = rotate_coordinates(
            df["pickup_latitude"], df["pickup_longitude"], ROTATION_ANGLE_RAD
        )
        df["pickup_latitude_rot"] = lat_rot
        df["pickup_longitude_rot"] = lon_rot

        # Rotate Dropoff
        lat_rot, lon_rot = rotate_coordinates(
            df["dropoff_latitude"], df["dropoff_longitude"], ROTATION_ANGLE_RAD
        )
        df["dropoff_latitude_rot"] = lat_rot
        df["dropoff_longitude_rot"] = lon_rot

        return df

    def add_physics_features(self, df):
        """
        Adds distance metrics including Haversine and Rotated Manhattan distance.
        Also adds distances to key landmarks.
        """
        # 1. Haversine Distance (Crow-flies)
        df["haversine_dist"] = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # 2. Rotated Manhattan Distance (Grid approximation)
        # Using the rotated coordinates generated in add_rotated_features
        df["manhattan_dist_rot"] = manhattan_distance(
            df["pickup_latitude_rot"],
            df["pickup_longitude_rot"],
            df["dropoff_latitude_rot"],
            df["dropoff_longitude_rot"],
        )

        # 3. Landmark Distances
        # Calculate distance from Pickup and Dropoff to each landmark
        for name, (lat, lon) in LANDMARKS.items():
            df[f"dist_pickup_{name}"] = haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"], lat, lon
            )
            df[f"dist_dropoff_{name}"] = haversine_distance(
                df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
            )

        return df

    def add_airport_flags(self, df):
        """
        Adds binary flags for pickups/dropoffs within airport bounding boxes.
        """
        for name, (min_lat, max_lat, min_lon, max_lon) in AIRPORT_BOXES.items():
            # Pickup Flag
            df[f"is_pickup_{name}"] = (
                (df["pickup_latitude"] >= min_lat)
                & (df["pickup_latitude"] <= max_lat)
                & (df["pickup_longitude"] >= min_lon)
                & (df["pickup_longitude"] <= max_lon)
            ).astype(int)

            # Dropoff Flag
            df[f"is_dropoff_{name}"] = (
                (df["dropoff_latitude"] >= min_lat)
                & (df["dropoff_latitude"] <= max_lat)
                & (df["dropoff_longitude"] >= min_lon)
                & (df["dropoff_longitude"] <= max_lon)
            ).astype(int)

        return df

    def process(self, df, name):
        """
        Master processing method. Applies all transformations and handles caching.

        Args:
            df (pd.DataFrame): Input DataFrame.
            name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.

        Returns:
            pd.DataFrame: Engineered DataFrame.
        """
        cache_path = os.path.join(WORKING_DIR, f"{name}_engineered.parquet")

        # Check cache
        if self.load_cached_data and os.path.exists(cache_path):
            print(f"Loading engineered data for {name} from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Engineering features for {name}...")

        # 1. Time Features
        df = self.add_time_features(df)

        # 2. Rotated Coordinates
        df = self.add_rotated_features(df)

        # 3. Physics Features (Distances)
        df = self.add_physics_features(df)

        # 4. Airport Flags
        df = self.add_airport_flags(df)

        # Drop original datetime to save memory and prevent model errors
        if "pickup_datetime" in df.columns:
            df = df.drop(columns=["pickup_datetime"])

        # Optimize memory usage
        df = reduce_mem_usage(df)

        # Save to cache
        print(f"Saving engineered data for {name} to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df
