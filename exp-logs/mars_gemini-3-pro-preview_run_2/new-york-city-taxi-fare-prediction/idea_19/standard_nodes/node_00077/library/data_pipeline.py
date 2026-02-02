import os
import gc
import pandas as pd
import numpy as np
from library.config import ProjectConfig
from library.utils import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    bearing,
    rotate_coordinates,
    calculate_geohash,
)
from library.stats_manager import StatsManager


class DataPipeline:
    def __init__(self):
        self.config = ProjectConfig
        self.stats_manager = StatsManager()
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self):
        """Returns the file paths for cached datasets."""
        return {
            "train": os.path.join(self.cache_dir, "processed_train_learner.parquet"),
            "val": os.path.join(self.cache_dir, "processed_val.parquet"),
            "test": os.path.join(self.cache_dir, "processed_test.parquet"),
        }

    def _feature_engineering(self, df):
        """
        Applies deterministic feature engineering:
        - Time features (Hour, Year, Month, Day, Weekday)
        - Distance features (Haversine, Manhattan)
        - Bearing
        - Rotated Coordinates
        - Geohashes (L5, L6, L7)
        """
        # Time features
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        dt = df["pickup_datetime"].dt
        df["hour"] = dt.hour.astype("int32")
        df["year"] = dt.year.astype("int32")
        df["month"] = dt.month.astype("int32")
        df["day"] = dt.day.astype("int32")
        df["weekday"] = dt.dayofweek.astype("int32")

        # Coordinate arrays
        plat = df["pickup_latitude"].values
        plon = df["pickup_longitude"].values
        dlat = df["dropoff_latitude"].values
        dlon = df["dropoff_longitude"].values

        # Distances
        df["distance_haversine"] = haversine_distance(plat, plon, dlat, dlon).astype(
            "float32"
        )
        df["distance_manhattan"] = manhattan_distance(plat, plon, dlat, dlon).astype(
            "float32"
        )

        # Bearing
        df["bearing"] = bearing(plat, plon, dlat, dlon).astype("float32")

        # Rotated Coordinates (aligned with NYC grid)
        plat_rot, plon_rot = rotate_coordinates(plat, plon)
        dlat_rot, dlon_rot = rotate_coordinates(dlat, dlon)
        df["pickup_latitude_rot"] = plat_rot.astype("float32")
        df["pickup_longitude_rot"] = plon_rot.astype("float32")
        df["dropoff_latitude_rot"] = dlat_rot.astype("float32")
        df["dropoff_longitude_rot"] = dlon_rot.astype("float32")

        # Geohashes (Required for StatsManager to attach priors)
        for l in self.config.GEOHASH_LEVELS:
            df[f"geohash_{l}"] = calculate_geohash(plat, plon, l)

        return df

    def get_data(self, load_cached=True):
        """
        Main entry point to retrieve processed data.
        Manages caching, subsampling, hygiene, and feature enrichment.
        """
        paths = self._get_cache_paths()

        # Check if all files exist in cache
        if load_cached and all(os.path.exists(p) for p in paths.values()):
            print("Loading processed data from cache...")
            try:
                train_df = pd.read_parquet(paths["train"])
                val_df = pd.read_parquet(paths["val"])
                test_df = pd.read_parquet(paths["test"])
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Error loading cache: {e}. Re-processing from scratch.")

        print("Processing data from scratch...")

        # 1. Compute/Load Global Wisdom Stats
        # This calls the StatsManager to load the full dataset, apply strict hygiene,
        # and compute the hierarchical priors.
        stats = self.stats_manager.compute_global_moments(load_cached=load_cached)

        # 2. Process Train (Learner Set)
        print("Processing Learner Train Set...")
        # Load raw metadata
        train_df = pd.read_parquet(self.config.TRAIN_PATH)

        # Subsample to fit in memory/time constraints
        if len(train_df) > self.config.TRAIN_SUBSAMPLE_SIZE:
            print(
                f"Subsampling training data to {self.config.TRAIN_SUBSAMPLE_SIZE} rows..."
            )
            train_df = train_df.sample(
                n=self.config.TRAIN_SUBSAMPLE_SIZE, random_state=self.config.SEED
            ).copy()

        # Sanitize (Clamp Coordinates)
        train_df = clamp_coordinates(train_df, inplace=True)

        # Feature Engineering
        train_df = self._feature_engineering(train_df)

        # Loose Filter (Learner Hygiene)
        # We keep high fares to learn the heavy tail, but remove impossible negatives/zeros
        # Also cap at MAX_FARE to avoid training on data errors (Cite Lesson 00017)
        train_df = train_df[
            (train_df["fare_amount"] >= self.config.LEARNER_MIN_FARE)
            & (train_df["fare_amount"] <= self.config.LEARNER_MAX_FARE)
        ].copy()

        # Enrich with Stats (Training Mode)
        # Since 'fare_amount' is present, StatsManager will assign folds and perform
        # Vectorized Subtraction (Global - Fold) to prevent leakage.
        train_df = self.stats_manager.compute_kfold_moments(train_df, stats)

        # Save to cache
        train_df.to_parquet(paths["train"])

        # 3. Process Validation Set
        print("Processing Validation Set...")
        val_df = pd.read_parquet(self.config.VAL_PATH)
        val_df = clamp_coordinates(val_df, inplace=True)

        # Apply Validation Hygiene (Cite Lesson 00071)
        # Filter out extreme outliers to ensure metric stability and comparability
        val_df = val_df[
            (val_df["fare_amount"] >= self.config.LEARNER_MIN_FARE)
            & (val_df["fare_amount"] <= self.config.VAL_MAX_FARE)
        ].copy()

        val_df = self._feature_engineering(val_df)

        # Enrich with Stats (Inference Mode)
        # For validation, we want to simulate test time (using full Global Wisdom).
        # We temporarily hide 'fare_amount' so StatsManager treats it as inference
        # and does NOT perform fold subtraction.
        y_val = val_df["fare_amount"].copy()
        val_df_no_target = val_df.drop(columns=["fare_amount"])
        val_df_enriched = self.stats_manager.compute_kfold_moments(
            val_df_no_target, stats
        )
        val_df_enriched["fare_amount"] = y_val
        val_df = val_df_enriched

        val_df.to_parquet(paths["val"])
        del val_df_no_target, val_df_enriched, y_val
        gc.collect()

        # 4. Process Test Set
        print("Processing Test Set...")
        test_df = pd.read_parquet(self.config.TEST_PATH)
        test_df = clamp_coordinates(test_df, inplace=True)
        test_df = self._feature_engineering(test_df)

        # Enrich with Stats (Inference Mode)
        # No 'fare_amount' present, so uses Global Wisdom stats directly.
        test_df = self.stats_manager.compute_kfold_moments(test_df, stats)

        test_df.to_parquet(paths["test"])

        # Return loaded dataframes
        return train_df, val_df, test_df
