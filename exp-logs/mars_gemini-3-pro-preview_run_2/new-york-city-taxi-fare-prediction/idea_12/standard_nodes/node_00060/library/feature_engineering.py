import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    haversine_distance,
    manhattan_distance,
    calculate_bearing,
    rotate_coordinates,
    clean_memory,
)


class FeatureEngineer:
    """
    Handles feature engineering for the Taxi Fare Prediction task.
    Generates physical features, temporal features, and spatial keys for Multi-View Encoding.
    Implements caching to disk to optimize runtime.
    """

    def __init__(self, config: Config):
        self.config = config

    def _add_time_features(self, df):
        """Extracts temporal features from pickup_datetime."""
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Handle potential " UTC" suffix if present in string (common in this dataset)
            # Checking first row is a fast heuristic
            if (
                not df.empty
                and isinstance(df["pickup_datetime"].iloc[0], str)
                and df["pickup_datetime"].iloc[0].endswith(" UTC")
            ):
                df["pickup_datetime"] = pd.to_datetime(
                    df["pickup_datetime"].str.slice(0, -4),
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce",
                )
            else:
                df["pickup_datetime"] = pd.to_datetime(
                    df["pickup_datetime"], errors="coerce"
                )

        dt = df["pickup_datetime"].dt
        df["hour"] = dt.hour.fillna(0).astype(np.int8)
        df["year"] = dt.year.fillna(0).astype(np.int16)
        df["month"] = dt.month.fillna(0).astype(np.int8)
        df["day"] = dt.day.fillna(0).astype(np.int8)
        df["weekday"] = dt.dayofweek.fillna(0).astype(np.int8)
        return df

    def _add_distance_features(self, df):
        """Computes basic Haversine distance."""
        df["dist_haversine"] = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        ).astype(np.float32)
        return df

    def _add_advanced_features(self, df):
        """Computes advanced physical features for training."""
        # Manhattan Distance
        df["dist_manhattan"] = manhattan_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        ).astype(np.float32)

        # Bearing
        df["bearing"] = calculate_bearing(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        ).astype(np.float32)

        # Rotated Coordinates (NYC Grid Alignment ~29 degrees)
        angle = 29.0
        p_rot_x, p_rot_y = rotate_coordinates(
            df["pickup_longitude"].values, df["pickup_latitude"].values, angle
        )
        d_rot_x, d_rot_y = rotate_coordinates(
            df["dropoff_longitude"].values, df["dropoff_latitude"].values, angle
        )

        df["pickup_rot_x"] = p_rot_x.astype(np.float32)
        df["pickup_rot_y"] = p_rot_y.astype(np.float32)
        df["dropoff_rot_x"] = d_rot_x.astype(np.float32)
        df["dropoff_rot_y"] = d_rot_y.astype(np.float32)

        # Coordinate Deltas
        df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(
            np.float32
        )
        df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
            np.float32
        )

        return df

    def _add_spatial_keys(self, df):
        """
        Generates discrete spatial keys for Multi-View Encoding.
        Simulates Geohash levels using decimal rounding.
        """
        # Round coordinates
        p_lat_fine = df["pickup_latitude"].round(self.config.PRECISION_FINE)
        p_lon_fine = df["pickup_longitude"].round(self.config.PRECISION_FINE)
        d_lat_fine = df["dropoff_latitude"].round(self.config.PRECISION_FINE)
        d_lon_fine = df["dropoff_longitude"].round(self.config.PRECISION_FINE)

        p_lat_coarse = df["pickup_latitude"].round(self.config.PRECISION_COARSE)
        p_lon_coarse = df["pickup_longitude"].round(self.config.PRECISION_COARSE)
        d_lat_coarse = df["dropoff_latitude"].round(self.config.PRECISION_COARSE)
        d_lon_coarse = df["dropoff_longitude"].round(self.config.PRECISION_COARSE)

        # Create String Keys for Aggregation/Joining
        # Using string concatenation is robust and handles the precision correctly

        # 1. Fine-Grained Route (~110m)
        df["key_fine"] = (
            p_lat_fine.astype(str)
            + "_"
            + p_lon_fine.astype(str)
            + "_"
            + d_lat_fine.astype(str)
            + "_"
            + d_lon_fine.astype(str)
        )

        # 2. Coarse-Grained Route (~1.1km)
        df["key_coarse"] = (
            p_lat_coarse.astype(str)
            + "_"
            + p_lon_coarse.astype(str)
            + "_"
            + d_lat_coarse.astype(str)
            + "_"
            + d_lon_coarse.astype(str)
        )

        # 3. Temporal-Spatial (Pickup Coarse + Hour)
        # Note: 'hour' must exist (computed in _add_time_features)
        df["key_temporal"] = (
            p_lat_coarse.astype(str)
            + "_"
            + p_lon_coarse.astype(str)
            + "_"
            + df["hour"].astype(str)
        )

        return df

    def process(self, df, cache_key, load_cached_data=True, is_background=False):
        """
        Main processing pipeline with caching.

        Args:
            df: Input DataFrame.
            cache_key: Unique identifier for the cache file (e.g. 'train_fg', 'background').
            load_cached_data: Whether to try loading from disk.
            is_background: If True, computes minimal features (keys + basic dist) for Knowledge Base.
                           If False, computes full feature set for training/inference.
        """
        cache_path = self.config.get_cache_path(f"featurized_{cache_key}.parquet")

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading featurized data from {cache_path}...")
            try:
                # If df is provided, we delete it to free memory before loading the cached version
                if df is not None:
                    del df
                    clean_memory()
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Processing from scratch.")

        # 2. Process from Scratch
        if df is None:
            raise ValueError(
                "DataFrame cannot be None if cache is missing or load_cached_data is False."
            )

        print(f"Featurizing {cache_key} (Background={is_background})...")

        # Temporal Features (Needed for key_temporal)
        df = self._add_time_features(df)

        # Spatial Keys (Needed for Aggregation or Joining)
        df = self._add_spatial_keys(df)

        # Distance Features (Haversine needed for Fare/Km stats in Background)
        df = self._add_distance_features(df)

        if not is_background:
            # Full Feature Set for Training/Test
            df = self._add_advanced_features(df)

        # 3. Save to Cache
        print(f"Saving featurized data to {cache_path}...")
        self.config.setup_dirs()
        df.to_parquet(cache_path, index=False)

        return df
