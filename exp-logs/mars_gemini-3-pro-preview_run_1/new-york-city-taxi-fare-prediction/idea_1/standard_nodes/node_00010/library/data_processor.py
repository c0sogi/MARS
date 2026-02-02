import os
import pandas as pd
import numpy as np
from library import config
from library import utils


class TaxiDataProcessor:
    def __init__(self):
        """
        Initializes the Data Processor with configurations and sets random seeds.
        """
        self.feature_config = config.FEATURE_CONFIG
        self.data_paths = config.DATA_PATHS
        self.working_dir = config.WORKING_DIR

        # Ensure reproducibility
        utils.set_seed(42)

    def _load_raw_data(self, split_name):
        """
        Loads the raw parquet file for the given split.
        """
        if split_name not in self.data_paths:
            raise ValueError(
                f"Unknown split: {split_name}. Available: {list(self.data_paths.keys())}"
            )

        path = self.data_paths[split_name]
        print(f"Loading raw data for {split_name} from {path}...")
        return pd.read_parquet(path)

    def clean_data(self, df):
        """
        Filters the dataset based on bounds defined in config.
        Removes rows with invalid coordinates, passenger counts, or fare amounts.
        """
        initial_len = len(df)
        bounds = self.feature_config["bounds"]

        # Filter by coordinates
        # We assume coordinates must be within valid Earth ranges and optionally NYC bounds if specified
        # Here we use the generic lat/lon bounds from config
        mask = (
            (df["pickup_latitude"].between(bounds["lat_min"], bounds["lat_max"]))
            & (df["pickup_longitude"].between(bounds["lon_min"], bounds["lon_max"]))
            & (df["dropoff_latitude"].between(bounds["lat_min"], bounds["lat_max"]))
            & (df["dropoff_longitude"].between(bounds["lon_min"], bounds["lon_max"]))
        )
        df = df[mask]

        # Filter by passenger count
        mask = df["passenger_count"].between(
            bounds["passenger_min"], bounds["passenger_max"]
        )
        df = df[mask]

        # Filter by fare amount if the column exists (Training/Validation data)
        target_col = self.feature_config["target_col"]
        if target_col in df.columns:
            mask = df[target_col].between(bounds["fare_min"], bounds["fare_max"])
            df = df[mask]

        print(f"Data cleaning removed {initial_len - len(df)} rows.")
        return df

    def engineer_features(self, df):
        """
        Generates temporal and spatial features.
        """
        dt_col = self.feature_config["datetime_col"]

        # Ensure datetime column is actually datetime objects
        # Using utc=True to handle potential timezone info in strings
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True)

        # Temporal Features
        df["pickup_hour"] = df[dt_col].dt.hour
        df["pickup_weekday"] = df[dt_col].dt.dayofweek
        df["pickup_month"] = df[dt_col].dt.month
        df["pickup_year"] = df[dt_col].dt.year

        # Monotonic Time Feature (Cite solution_lesson_node_00005)
        # Captures global trends like inflation better than decomposed year/month
        df["months_since_start"] = (df["pickup_year"] - 2009) * 12 + (
            df["pickup_month"] - 1
        )

        # Spatial Features
        # Haversine Distance
        df["distance_haversine"] = utils.haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Manhattan Distance
        df["distance_manhattan"] = utils.manhattan_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Absolute Coordinate Differences
        df["abs_diff_lat"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
        df["abs_diff_lon"] = np.abs(df["pickup_longitude"] - df["dropoff_longitude"])

        # Airport and Landmark Distances (Cite solution_lesson_node_00001, solution_lesson_node_00009)
        # Adding granular anchors (Penn, Grand Central, WTC) to capture micro-regional pricing
        landmarks = [
            ("jfk", utils.JFK_COORD),
            ("lga", utils.LGA_COORD),
            ("ewr", utils.EWR_COORD),
            ("penn", utils.PENN_STATION_COORD),
            ("grand", utils.GRAND_CENTRAL_COORD),
            ("wtc", utils.WTC_COORD),
        ]

        for name, (lat, lon) in landmarks:
            df[f"pickup_dist_{name}"] = utils.haversine_distance(
                df["pickup_latitude"].values, df["pickup_longitude"].values, lat, lon
            )
            df[f"dropoff_dist_{name}"] = utils.haversine_distance(
                df["dropoff_latitude"].values, df["dropoff_longitude"].values, lat, lon
            )

        # Distance to NYC Center
        center_lat, center_lon = utils.NYC_CENTER_COORD
        df["pickup_dist_center"] = utils.haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            center_lat,
            center_lon,
        )
        df["dropoff_dist_center"] = utils.haversine_distance(
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
            center_lat,
            center_lon,
        )

        # Bearing
        df["bearing"] = utils.calculate_bearing(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Rotated Coordinates (45 degrees) to align with Manhattan grid structure
        # (Cite solution_lesson_node_00001: Overcoming orthogonal split limitations)
        df["pickup_rot_sum"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["pickup_rot_diff"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_rot_sum"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        df["dropoff_rot_diff"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        return df

    def process_data(self, split_name, load_cached_data=True, debug_sample_size=None):
        """
        Main pipeline to load, clean, and engineer features.
        Handles caching to disk to speed up subsequent runs.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from ./working/idea_1/.
            debug_sample_size (int, optional): If set, samples the raw data before processing.
                                               Note: This will save the sampled version to cache.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        cache_filename = f"processed_{split_name}.parquet"
        cache_path = os.path.join(self.working_dir, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached processed data for '{split_name}' from {cache_path}..."
            )
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache ({e}). Reprocessing from scratch.")

        # 2. Process from scratch
        print(f"Processing '{split_name}' data from scratch...")

        # Load Raw
        df = self._load_raw_data(split_name)

        # Debug Sampling
        if debug_sample_size is not None and len(df) > debug_sample_size:
            print(f"Sampling {debug_sample_size} rows for debugging...")
            df = df.sample(n=debug_sample_size, random_state=42).reset_index(drop=True)

        # Clean Data
        # IMPORTANT: Do not drop rows from the test set, as we need to predict for every key.
        if split_name != "test":
            df = self.clean_data(df)
        else:
            print("Skipping data cleaning (row dropping) for test set.")

        # Engineer Features
        df = self.engineer_features(df)

        # 3. Save to cache
        print(f"Saving processed data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df
