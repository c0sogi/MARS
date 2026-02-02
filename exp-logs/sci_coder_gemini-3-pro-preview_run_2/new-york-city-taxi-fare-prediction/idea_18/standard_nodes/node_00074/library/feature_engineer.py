import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import haversine_distance, manhattan_distance
from library.stats_computer import HierarchicalStatsEngine


class FeatureEngineer:
    """
    Orchestrates the feature engineering pipeline for the Multi-Moment Hierarchical
    Dual-Hygiene Gradient Boosting strategy.

    Responsibilities:
    1. Generate explicit geometric and temporal features.
    2. Integrate with HierarchicalStatsEngine to generate statistical priors.
    3. Manage caching of processed datasets.
    """

    def __init__(self):
        self.stats_engine = HierarchicalStatsEngine()

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal features from the pickup_datetime timestamp.
        """
        df = df.copy()

        # Convert to datetime if strictly string/object
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Optimization: Strip ' UTC' suffix if present to speed up parsing
            # We check the first element to determine if stripping is needed
            if not df["pickup_datetime"].empty:
                first_val = df["pickup_datetime"].iloc[0]
                if isinstance(first_val, str) and first_val.endswith(" UTC"):
                    df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, -4)

            df["pickup_datetime"] = pd.to_datetime(
                df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )

        # Extract cyclic and ordinal components
        df["hour"] = df["pickup_datetime"].dt.hour
        df["year"] = df["pickup_datetime"].dt.year
        df["month"] = df["pickup_datetime"].dt.month
        df["day"] = df["pickup_datetime"].dt.day
        df["weekday"] = df["pickup_datetime"].dt.dayofweek

        # Additional aggregations
        df["quarter_of_day"] = df["hour"] // 6

        return df

    def _add_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates geometric features including distances, coordinate differences,
        and rotated coordinates to capture grid structures.
        """
        df = df.copy()

        p_lat = df["pickup_latitude"]
        p_lon = df["pickup_longitude"]
        d_lat = df["dropoff_latitude"]
        d_lon = df["dropoff_longitude"]

        # 1. Raw Coordinate Differences
        df["delta_lat"] = d_lat - p_lat
        df["delta_lon"] = d_lon - p_lon
        df["abs_diff_lat"] = df["delta_lat"].abs()
        df["abs_diff_lon"] = df["delta_lon"].abs()

        # 2. Distances (using library utils)
        df["haversine_dist"] = haversine_distance(p_lat, p_lon, d_lat, d_lon)
        df["manhattan_dist"] = manhattan_distance(p_lat, p_lon, d_lat, d_lon)

        # 3. Rotated Coordinates (45 degrees)
        # In a grid city like NYC, rotating coordinates by 45 degrees can align
        # features with the street grid.
        # x' = x + y, y' = x - y (scaled rotation)
        df["pickup_rot_sum"] = p_lat + p_lon
        df["pickup_rot_diff"] = p_lat - p_lon
        df["dropoff_rot_sum"] = d_lat + d_lon
        df["dropoff_rot_diff"] = d_lat - d_lon

        return df

    def process_train_data(
        self,
        wisdom_df: pd.DataFrame,
        learner_df: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Engineers features for the training set.

        Steps:
        1. Fit the StatsEngine on the Wisdom Set (Global Priors).
        2. Generate basic geometric/temporal features for the Learner Set.
        3. Apply K-Fold Vectorized Subtraction to the Learner Set to generate
           unbiased statistical features.

        Args:
            wisdom_df: High-quality data for generating priors.
            learner_df: Subsampled data for training the model.
            load_cached_data: Whether to use cached output.

        Returns:
            pd.DataFrame: Fully featurized training data.
        """
        cache_path = Config.CACHE_PROCESSED_TRAIN

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached processed training data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Generating features for Training Data...")

        # 1. Fit Stats Engine
        # This computes and caches the global stats from the Wisdom Set.
        self.stats_engine.fit(wisdom_df, load_cached_data=load_cached_data)

        # 2. Basic Features
        learner_df = self._add_temporal_features(learner_df)
        learner_df = self._add_geometric_features(learner_df)

        # 3. Statistical Features (K-Fold Subtraction)
        # This generates the Multi-Moment features (Mean, Std, Count) while preventing leakage.
        learner_df = self.stats_engine.transform_train(learner_df)

        # Cache result
        print(f"Caching processed training data to {cache_path}...")
        learner_df.to_parquet(cache_path, index=False)

        return learner_df

    def process_validation_data(
        self,
        val_df: pd.DataFrame,
        wisdom_df: pd.DataFrame = None,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Engineers features for the validation set.
        Uses Global Stats Mapping (no subtraction).
        """
        cache_path = Config.CACHE_PROCESSED_VAL

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached processed validation data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Generating features for Validation Data...")

        # Ensure engine is fitted with global stats
        if not self.stats_engine.global_stats:
            if wisdom_df is not None:
                self.stats_engine.fit(wisdom_df, load_cached_data=True)
            else:
                # Attempt to load stats from disk if wisdom_df is not provided
                # This assumes stats were computed in a previous step
                self.stats_engine.fit(pd.DataFrame(), load_cached_data=True)

        # Basic Features
        val_df = self._add_temporal_features(val_df)
        val_df = self._add_geometric_features(val_df)

        # Statistical Features (Global Mapping)
        val_df = self.stats_engine.transform_test(val_df)

        # Cache
        print(f"Caching processed validation data to {cache_path}...")
        val_df.to_parquet(cache_path, index=False)

        return val_df

    def process_test_data(
        self,
        test_df: pd.DataFrame,
        wisdom_df: pd.DataFrame = None,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Engineers features for the test set.
        Uses Global Stats Mapping (no subtraction).
        """
        cache_path = Config.CACHE_PROCESSED_TEST

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached processed test data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Generating features for Test Data...")

        # Ensure engine is fitted
        if not self.stats_engine.global_stats:
            if wisdom_df is not None:
                self.stats_engine.fit(wisdom_df, load_cached_data=True)
            else:
                self.stats_engine.fit(pd.DataFrame(), load_cached_data=True)

        # Basic Features
        test_df = self._add_temporal_features(test_df)
        test_df = self._add_geometric_features(test_df)

        # Statistical Features (Global Mapping)
        test_df = self.stats_engine.transform_test(test_df)

        # Cache
        print(f"Caching processed test data to {cache_path}...")
        test_df.to_parquet(cache_path, index=False)

        return test_df
