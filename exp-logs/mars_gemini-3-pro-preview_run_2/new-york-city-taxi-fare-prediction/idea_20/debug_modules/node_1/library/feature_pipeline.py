import os
import pandas as pd
import numpy as np
from library import config
from library import utils
from library.stats_computer import StatsEngine


class FeatureGenerator:
    """
    Orchestrates the feature generation pipeline.
    Combines hierarchical statistical moments from StatsEngine with
    explicit geometric and temporal features.
    """

    def __init__(self):
        self.stats_engine = StatsEngine()
        self.cache_dir = config.CACHE_DIR
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates explicit geometric and temporal features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added features.
        """
        df = df.copy()

        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        # 1. Temporal Features
        df["hour"] = df["pickup_datetime"].dt.hour
        df["year"] = df["pickup_datetime"].dt.year
        df["weekday"] = df["pickup_datetime"].dt.dayofweek
        # Month/Day might be useful for seasonality, but Year/Hour are primary
        df["month"] = df["pickup_datetime"].dt.month

        # 2. Geometric Features
        # Haversine Distance (Great Circle)
        df["dist_haversine"] = utils.calculate_haversine(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Manhattan Distance (L1 Norm approximation)
        df["dist_manhattan"] = utils.calculate_manhattan(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # 3. Rotated Coordinates
        # Useful for tree-based models to capture diagonal boundaries (e.g., street grids)
        # Rotation by 45 degrees: x' = x + y, y' = x - y
        df["pickup_rot_1"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["pickup_rot_2"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_rot_1"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        df["dropoff_rot_2"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        # 4. Directional Deltas
        df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
        df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]

        return df

    def process(
        self,
        learner_df: pd.DataFrame,
        wisdom_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Main execution method for the feature pipeline.

        1. Checks cache for pre-computed feature matrices.
        2. If missing, computes Global Stats via StatsEngine.
        3. Enriches Learner (Train), Val, and Test sets with moments.
           - Applies Conditional Vectorized Subtraction for Learner set.
        4. Adds base geometric/temporal features.
        5. Caches the results.
        6. Returns X, y matrices for training.

        Args:
            learner_df: Training data (subsampled).
            wisdom_df: High-quality data for global priors.
            val_df: Validation data.
            test_df: Test data.
            load_cached_data: Boolean flag to use cache.

        Returns:
            X_train, y_train, X_val, y_val, X_test, test_keys
        """
        # Define Cache Paths
        path_train = os.path.join(self.cache_dir, "featurized_train.parquet")
        path_val = os.path.join(self.cache_dir, "featurized_val.parquet")
        path_test = os.path.join(self.cache_dir, "featurized_test.parquet")

        # Check Cache
        if (
            load_cached_data
            and os.path.exists(path_train)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
        ):
            print("Loading featurized data from cache...")
            train_final = pd.read_parquet(path_train)
            val_final = pd.read_parquet(path_val)
            test_final = pd.read_parquet(path_test)
        else:
            print("Generating features from scratch...")

            # 1. Compute Global Statistical Priors (Moments)
            # This uses the Wisdom set to create robust Mean/Std maps
            global_stats = self.stats_engine.compute_global_stats(
                wisdom_df, load_cached_data=load_cached_data
            )

            # 2. Enrich Datasets with Hierarchical Moments

            # Train Set: Use mode='train' to trigger Conditional Vectorized Subtraction.
            # This ensures that the model doesn't see the target of the specific row
            # inside the aggregate feature (Leakage Prevention).
            print("Enriching Training Data (Learner)...")
            train_enriched = self.stats_engine.enrich_data(
                learner_df, global_stats, mode="train"
            )

            # Val/Test Sets: Use mode='test'.
            # Direct mapping of global priors. No subtraction needed as these rows
            # were not used to build the global stats (strictly separated).
            print("Enriching Validation Data...")
            val_enriched = self.stats_engine.enrich_data(
                val_df, global_stats, mode="test"
            )

            print("Enriching Test Data...")
            test_enriched = self.stats_engine.enrich_data(
                test_df, global_stats, mode="test"
            )

            # 3. Add Base Features
            print("Adding Base Geometric/Temporal Features...")
            train_final = self._add_base_features(train_enriched)
            val_final = self._add_base_features(val_enriched)
            test_final = self._add_base_features(test_enriched)

            # 4. Save to Cache
            print("Saving featurized data to cache...")
            train_final.to_parquet(path_train, index=False)
            val_final.to_parquet(path_val, index=False)
            test_final.to_parquet(path_test, index=False)

        # ---------------------------------------------------------
        # Prepare Output Matrices
        # ---------------------------------------------------------
        # Define columns to drop (identifiers, targets, raw timestamps)
        ignore_cols = ["key", "fare_amount", "pickup_datetime"]

        # Identify feature columns
        feature_cols = [c for c in train_final.columns if c not in ignore_cols]
        print(f"Final Feature List ({len(feature_cols)}): {feature_cols}")

        # Construct X and y
        X_train = train_final[feature_cols]
        y_train = train_final["fare_amount"]

        X_val = val_final[feature_cols]
        y_val = val_final["fare_amount"]

        X_test = test_final[feature_cols]
        test_keys = test_final["key"]

        return X_train, y_train, X_val, y_val, X_test, test_keys
