import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    PROCESSED_TEST_PATH,
    LEARNER_CRITERIA,
    NYC_BOUNDING_BOX,
    MODEL_FEATURES,
)
import library.config as config
from library.spatial_ops import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    add_rotated_coordinates,
)
from library.statistics_manager import RouteStatCalculator


class PipelineProcessor:
    """
    Orchestrates the data processing pipeline for the Variance-Aware Dual-Hygiene strategy.
    Handles data loading, spatial clamping, feature engineering, and the attachment
    of distributional priors via the RouteStatCalculator.
    """

    def __init__(self):
        self.stats_calculator = RouteStatCalculator()
        self.temporal_stats = None

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies physics, temporal, and spatial feature engineering.
        """
        df = df.copy()

        # 1. Temporal Features
        if "pickup_datetime" in df.columns:
            # Efficiently convert to datetime, handling potential ' UTC' suffix
            if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
                # Fast path: check first element
                first_val = df["pickup_datetime"].iloc[0]
                if isinstance(first_val, str) and first_val.endswith(" UTC"):
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
            df["hour"] = dt.hour
            df["weekday"] = dt.dayofweek
            df["year"] = dt.year

        # 2. Physics / Distance Features
        # Ensure coordinates are float
        cols = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]
        for c in cols:
            df[c] = df[c].astype(float)

        df["dist_haversine"] = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        df["dist_manhattan"] = manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # 3. Rotated Coordinates (Orthogonal Spatial Features)
        df = add_rotated_coordinates(df)

        return df

    def _apply_temporal_rate(
        self, df: pd.DataFrame, is_train: bool = False
    ) -> pd.DataFrame:
        """
        Calculates or applies the 'temporal_fare_rate' (Mean Fare per Km by Hour).
        For training data, it computes the stats. For val/test, it maps them.
        """
        df = df.copy()

        if is_train:
            # Compute rates on the training set (Learner Hygiene)
            # Filter for valid distances to avoid division by zero
            valid_mask = (df["dist_haversine"] > 0.1) & (df["fare_amount"] > 0)
            temp_df = df[valid_mask].copy()
            temp_df["rate"] = temp_df["fare_amount"] / temp_df["dist_haversine"]

            # Group by hour and compute mean
            self.temporal_stats = temp_df.groupby("hour")["rate"].mean().to_dict()

        # Apply mapping
        if self.temporal_stats is None:
            # Fallback (should not happen if train is processed/loaded first)
            # Using a reasonable default of $2.50/km
            print("Warning: Temporal stats not found. Using default.")
            df["temporal_fare_rate"] = 2.5
        else:
            global_mean = np.mean(list(self.temporal_stats.values()))
            df["temporal_fare_rate"] = (
                df["hour"].map(self.temporal_stats).fillna(global_mean)
            )

        return df

    def process_data(self, load_cached_data: bool = True):
        """
        Main pipeline execution method.

        Args:
            load_cached_data: If True, attempts to load processed files from disk.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)

        # ---------------------------------------------------------
        # 1. Global Spatial Statistics (Wisdom)
        # ---------------------------------------------------------
        # This aggregates stats from the FULL 55M dataset (or loads cache)
        global_stats = self.stats_calculator.aggregate_global_stats(
            load_cached_data=load_cached_data
        )

        # ---------------------------------------------------------
        # 2. Process Training Data
        # ---------------------------------------------------------
        if load_cached_data and os.path.exists(PROCESSED_TRAIN_PATH):
            print(f"Loading processed training data from {PROCESSED_TRAIN_PATH}...")
            train_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
            # Re-populate temporal stats from the loaded training data for consistency
            self._apply_temporal_rate(train_df, is_train=True)
        else:
            print("Processing training data from scratch...")
            # Load raw metadata
            train_df = pd.read_parquet(TRAIN_DATA_PATH)

            # Subsample to stable size
            # Cite debug_lesson_7: Access config dynamically to pick up runtime patches
            if len(train_df) > config.TRAIN_SAMPLE_SIZE:
                train_df = train_df.sample(n=config.TRAIN_SAMPLE_SIZE, random_state=42)

            # Apply Learner Criteria (Hygiene)
            mask = (train_df["fare_amount"] >= LEARNER_CRITERIA["min_fare"]) & (
                train_df["fare_amount"] <= LEARNER_CRITERIA["max_fare"]
            )
            train_df = train_df[mask].copy()

            # Spatial Clamping
            train_df = clamp_coordinates(train_df)

            # Feature Engineering
            train_df = self._feature_engineering(train_df)

            # Attach Distributional Priors (Vectorized Subtraction / LOO)
            train_df = self.stats_calculator.retrieve_and_subtract_priors(
                train_df, global_stats
            )

            # Attach Temporal Rate
            train_df = self._apply_temporal_rate(train_df, is_train=True)

            # Save to Cache
            train_df.to_parquet(PROCESSED_TRAIN_PATH)

        # ---------------------------------------------------------
        # 3. Process Validation Data
        # ---------------------------------------------------------
        if load_cached_data and os.path.exists(PROCESSED_VAL_PATH):
            print(f"Loading processed validation data from {PROCESSED_VAL_PATH}...")
            val_df = pd.read_parquet(PROCESSED_VAL_PATH)
        else:
            print("Processing validation data from scratch...")
            val_df = pd.read_parquet(VAL_DATA_PATH)

            # Spatial Clamping
            val_df = clamp_coordinates(val_df)

            # Feature Engineering
            val_df = self._feature_engineering(val_df)

            # Attach Distributional Priors (Direct Mapping)
            # Validation set is disjoint from the Global Stats source (Train), so no subtraction needed.
            val_df = self.stats_calculator.retrieve_priors_test(val_df, global_stats)

            # Attach Temporal Rate
            val_df = self._apply_temporal_rate(val_df, is_train=False)

            # Save to Cache
            val_df.to_parquet(PROCESSED_VAL_PATH)

        # ---------------------------------------------------------
        # 4. Process Test Data
        # ---------------------------------------------------------
        if load_cached_data and os.path.exists(PROCESSED_TEST_PATH):
            print(f"Loading processed test data from {PROCESSED_TEST_PATH}...")
            test_df = pd.read_parquet(PROCESSED_TEST_PATH)
        else:
            print("Processing test data from scratch...")
            test_df = pd.read_parquet(TEST_DATA_PATH)

            # Spatial Clamping
            test_df = clamp_coordinates(test_df)

            # Feature Engineering
            test_df = self._feature_engineering(test_df)

            # Attach Distributional Priors (Direct Mapping)
            test_df = self.stats_calculator.retrieve_priors_test(test_df, global_stats)

            # Attach Temporal Rate
            test_df = self._apply_temporal_rate(test_df, is_train=False)

            # Save to Cache
            test_df.to_parquet(PROCESSED_TEST_PATH)

        return train_df, val_df, test_df
