import os
import numpy as np
import pandas as pd
import gc
from library.utils import haversine_distance, manhattan_distance, rotate_coordinates
from library.config import WORKING_DIR


class FeatureEngineer:
    """
    Handles the generation of inductive bias features for the NYC Taxi Fare Prediction task.
    Includes datetime feature extraction and physics-based feature engineering.
    """

    def __init__(self):
        # Define cache directory within the working directory
        self.cache_dir = os.path.join(WORKING_DIR, "feature_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def add_datetime_features(self, df):
        """
        Extracts temporal features from the 'pickup_datetime' column.

        Args:
            df (pd.DataFrame): Input dataframe containing 'pickup_datetime'.

        Returns:
            pd.DataFrame: Dataframe with added temporal features.
        """
        print("Engineering Datetime Features...")

        # Ensure pickup_datetime is available
        if "pickup_datetime" not in df.columns:
            print(
                "Warning: 'pickup_datetime' column not found. Skipping datetime features."
            )
            return df

        # Convert to datetime object
        # Check if it's already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Clean string if necessary (remove ' UTC' suffix for speed)
            # Check first value to decide strategy
            first_val = df["pickup_datetime"].iloc[0] if not df.empty else ""
            if isinstance(first_val, str) and first_val.endswith(" UTC"):
                temp_series = df["pickup_datetime"].str.slice(0, -4)
            else:
                temp_series = df["pickup_datetime"]

            df["pickup_datetime"] = pd.to_datetime(
                temp_series, format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )

        # Extract components
        df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("int32")
        df["pickup_year"] = df["pickup_datetime"].dt.year.astype("int32")
        df["pickup_month"] = df["pickup_datetime"].dt.month.astype("int32")
        df["pickup_day"] = df["pickup_datetime"].dt.day.astype("int32")
        df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek.astype("int32")

        # Drop original datetime column to finalize feature matrix for XGBoost
        df.drop(columns=["pickup_datetime"], inplace=True)

        return df

    def add_physics_features(self, df):
        """
        Calculates physics-based features including distances and rotated coordinates.

        Args:
            df (pd.DataFrame): Input dataframe with coordinate columns.

        Returns:
            pd.DataFrame: Dataframe with added physics features.
        """
        print("Engineering Physics Features...")

        # Coordinates
        p_lat = df["pickup_latitude"].values
        p_lon = df["pickup_longitude"].values
        d_lat = df["dropoff_latitude"].values
        d_lon = df["dropoff_longitude"].values

        # 1. Haversine Distance (Great Circle)
        df["dist_haversine"] = haversine_distance(p_lat, p_lon, d_lat, d_lon).astype(
            np.float32
        )

        # 2. Manhattan Distance (L1 Norm in Degrees)
        df["dist_manhattan"] = manhattan_distance(p_lat, p_lon, d_lat, d_lon).astype(
            np.float32
        )

        # 3. Rotated Coordinates (Align with NYC Grid ~29 degrees)
        # We rotate both pickup and dropoff points
        rot_angle = 29.0
        p_rot_lat, p_rot_lon = rotate_coordinates(p_lat, p_lon, angle_degrees=rot_angle)
        d_rot_lat, d_rot_lon = rotate_coordinates(d_lat, d_lon, angle_degrees=rot_angle)

        df["pickup_rot_lat"] = p_rot_lat.astype(np.float32)
        df["pickup_rot_lon"] = p_rot_lon.astype(np.float32)
        df["dropoff_rot_lat"] = d_rot_lat.astype(np.float32)
        df["dropoff_rot_lon"] = d_rot_lon.astype(np.float32)

        return df

    def process_features(self, df, cache_key, load_cached_data=True):
        """
        Orchestrates the feature engineering pipeline with caching.

        Args:
            df (pd.DataFrame): Input dataframe (raw or spatially processed).
            cache_key (str): Unique identifier for the dataset (e.g., 'train_subsample', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed dataframe with all features.
        """
        # Construct cache path
        cache_path = os.path.join(self.cache_dir, f"{cache_key}_features.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            # We return the cached dataframe directly, ignoring input df
            return pd.read_parquet(cache_path)

        # 2. Compute Features
        print(f"Generating features for {cache_key}...")

        # Apply Datetime Features
        df = self.add_datetime_features(df)

        # Apply Physics Features
        df = self.add_physics_features(df)

        # 3. Save Cache
        print(f"Saving features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        # Garbage collection to free up memory
        gc.collect()

        return df
