import os
import numpy as np
import pandas as pd
from library.config import PATH_CONFIG, CACHE_DIR, BOUNDING_BOX, HUB_LOCATIONS, SEED


class TaxiDataProcessor:
    def __init__(self):
        self.R_EARTH_KM = 6371.0

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Vectorized Haversine distance calculation.
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return self.R_EARTH_KM * c

    def _manhattan_distance(self, lat1, lon1, lat2, lon2):
        """
        Manhattan distance approximation (L1 norm of coordinates).
        """
        return np.abs(lat2 - lat1) + np.abs(lon2 - lon1)

    def clamp_coordinates(self, df):
        """
        Clamps coordinate columns to the bounding box defined in config.
        """
        df["pickup_latitude"] = df["pickup_latitude"].clip(
            BOUNDING_BOX["lat_min"], BOUNDING_BOX["lat_max"]
        )
        df["pickup_longitude"] = df["pickup_longitude"].clip(
            BOUNDING_BOX["lon_min"], BOUNDING_BOX["lon_max"]
        )
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(
            BOUNDING_BOX["lat_min"], BOUNDING_BOX["lat_max"]
        )
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(
            BOUNDING_BOX["lon_min"], BOUNDING_BOX["lon_max"]
        )
        return df

    def add_distance_features(self, df):
        """
        Adds Haversine and Manhattan distance features.
        """
        df["dist_haversine"] = self._haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        df["dist_manhattan"] = self._manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Bearing/Direction could be added here, but distance is the primary requirement
        # Adding a simple coordinate difference feature set
        df["abs_diff_longitude"] = (
            df["dropoff_longitude"] - df["pickup_longitude"]
        ).abs()
        df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

        # Add rotated coordinates (45 degrees) to help tree models with diagonal distances
        # Rotation: x' = (x + y) / sqrt(2), y' = (x - y) / sqrt(2)
        # This aligns the coordinate system with the Manhattan street grid (roughly)
        angle = 0.7071  # 1 / sqrt(2)
        df["pickup_rot45_lat"] = (
            df["pickup_latitude"] + df["pickup_longitude"]
        ) * angle
        df["pickup_rot45_lon"] = (
            df["pickup_latitude"] - df["pickup_longitude"]
        ) * angle
        df["dropoff_rot45_lat"] = (
            df["dropoff_latitude"] + df["dropoff_longitude"]
        ) * angle
        df["dropoff_rot45_lon"] = (
            df["dropoff_latitude"] - df["dropoff_longitude"]
        ) * angle

        return df

    def add_hub_features(self, df):
        """
        Adds distance features to major hubs.
        """
        for hub_name, (hub_lat, hub_lon) in HUB_LOCATIONS.items():
            # Distance from pickup to hub
            df[f"dist_pickup_{hub_name}"] = self._haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"], hub_lat, hub_lon
            )
            # Distance from dropoff to hub
            df[f"dist_dropoff_{hub_name}"] = self._haversine_distance(
                df["dropoff_latitude"], df["dropoff_longitude"], hub_lat, hub_lon
            )
        return df

    def extract_time_features(self, df):
        """
        Extracts temporal features from pickup_datetime.
        """
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Handle " UTC" suffix if present (common in this dataset)
            # We assume the format is relatively standard based on metadata analysis
            try:
                df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
            except:
                # Fallback for string cleaning if direct conversion fails
                df["pickup_datetime"] = pd.to_datetime(
                    df["pickup_datetime"].astype(str).str.replace(" UTC", ""),
                    format="mixed",
                )

        df["hour"] = df["pickup_datetime"].dt.hour
        df["day"] = df["pickup_datetime"].dt.day
        df["month"] = df["pickup_datetime"].dt.month
        df["year"] = df["pickup_datetime"].dt.year
        df["weekday"] = df["pickup_datetime"].dt.dayofweek

        # Drop the original datetime object as it's not needed for XGBoost
        df = df.drop(columns=["pickup_datetime"])
        return df

    def _process_dataframe(self, df):
        """
        Applies the full processing pipeline to a dataframe.
        """
        # 1. Clamp Coordinates
        df = self.clamp_coordinates(df)

        # 2. Feature Engineering
        df = self.add_distance_features(df)
        df = self.add_hub_features(df)
        df = self.extract_time_features(df)

        # Drop key column if present, as it's not a feature (keep it only if needed for submission,
        # but usually we handle X and y. For test set, we might need to preserve order,
        # but the cache will store the processed dataframe including key if we don't drop it.
        # We will keep 'key' in the dataframe for tracking, but it should be dropped before training.)

        return df

    def get_processed_data(self, dataset_type, load_cached_data=True):
        """
        Retrieves processed data, using cache if available.

        Args:
            dataset_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        cache_path = os.path.join(CACHE_DIR, f"{dataset_type}_processed.parquet")

        # 1. Try to load from cache
        # Cite debug_lesson_1: Explicitly Invalidate Caches When Modifying Data Logic.
        if load_cached_data and os.path.exists(cache_path) and dataset_type != "test":
            try:
                # print(f"Loading cached {dataset_type} data from {cache_path}...")
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from scratch
        # print(f"Processing {dataset_type} data from scratch...")

        # Map dataset_type to raw file path
        if dataset_type == "train":
            raw_path = PATH_CONFIG["train_data"]
        elif dataset_type == "val":
            raw_path = PATH_CONFIG["val_data"]
        elif dataset_type == "test":
            raw_path = PATH_CONFIG["test_data"]
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        if raw_path.endswith(".csv"):
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_parquet(raw_path)

        # Apply processing
        df = self._process_dataframe(df)

        # 3. Save to cache
        # print(f"Saving processed {dataset_type} data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df
