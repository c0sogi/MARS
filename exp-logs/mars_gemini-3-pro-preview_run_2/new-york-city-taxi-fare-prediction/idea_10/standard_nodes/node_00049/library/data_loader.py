import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    GRID_PRECISION,
    SUBSAMPLE_SIZE,
    MAX_FARE,
    MAX_FARE_PER_KM,
    RANDOM_SEED,
    NYC_MIN_LON,
    NYC_MAX_LON,
    NYC_MIN_LAT,
    NYC_MAX_LAT,
)
from library.feature_engineering import (
    haversine_distance,
    clamp_coordinates,
    extract_datetime_features,
    add_distance_features,
    discretize_coordinates,
)


class GlobalRouteStats:
    """
    Stage 1: Global Knowledge Base.
    Processes the full dataset to build a physics-consistent prior (base margin).
    """

    def __init__(self):
        self.stats_cache_path = os.path.join(WORKING_DIR, "global_route_stats.parquet")
        self.meta_cache_path = os.path.join(WORKING_DIR, "global_meta.npy")

    def compute_and_cache(self, load_cached_data=True):
        """
        Computes or loads the global route statistics.
        """
        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(self.stats_cache_path)
            and os.path.exists(self.meta_cache_path)
        ):
            print(f"Loading cached Global Route Stats from {self.stats_cache_path}")
            stats_df = pd.read_parquet(self.stats_cache_path)
            meta = np.load(self.meta_cache_path, allow_pickle=True).item()
            return stats_df, meta["global_per_km"]

        print("Computing Global Route Stats from scratch (Stage 1)...")
        # 2. Load full training data
        # Using pyarrow engine for speed on large files
        df = pd.read_parquet(TRAIN_DATA_PATH, engine="pyarrow")

        # 3. Physics-Consistent Filtering
        # Calculate distance for filtering
        # We only need haversine for filtering here
        dist = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Filter logic
        # Remove garbage outliers (e.g. $93k) and impossible fare/distance ratios
        # Avoid division by zero with a small epsilon
        fare_per_km = df["fare_amount"] / (dist + 1e-6)

        mask = (
            (df["fare_amount"] > 0)
            & (df["fare_amount"] <= MAX_FARE)
            & (fare_per_km <= MAX_FARE_PER_KM)
        )
        df_clean = df.loc[mask].copy()

        # Add distance to clean df for global scalar calculation
        df_clean["distance_haversine"] = dist[mask]

        # 4. Calculate Global Scalar (Fallback)
        total_fare = df_clean["fare_amount"].sum()
        total_dist = df_clean["distance_haversine"].sum()
        global_per_km = total_fare / (total_dist + 1e-6)

        # 5. High-Resolution Discretization
        # Create grid keys
        df_clean["p_lat_g"] = discretize_coordinates(
            df_clean["pickup_latitude"], GRID_PRECISION
        )
        df_clean["p_lon_g"] = discretize_coordinates(
            df_clean["pickup_longitude"], GRID_PRECISION
        )
        df_clean["d_lat_g"] = discretize_coordinates(
            df_clean["dropoff_latitude"], GRID_PRECISION
        )
        df_clean["d_lon_g"] = discretize_coordinates(
            df_clean["dropoff_longitude"], GRID_PRECISION
        )

        # 6. Global Aggregation
        # Group by grid keys and compute sum/count
        stats_df = (
            df_clean.groupby(["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"])[
                "fare_amount"
            ]
            .agg(["sum", "count"])
            .reset_index()
        )
        stats_df.rename(
            columns={"sum": "global_sum", "count": "global_count"}, inplace=True
        )

        # 7. Save to cache
        print(f"Saving Global Route Stats to {self.stats_cache_path}")
        stats_df.to_parquet(self.stats_cache_path)
        np.save(self.meta_cache_path, {"global_per_km": global_per_km})

        return stats_df, global_per_km


class DatasetBuilder:
    """
    Stage 2: Training Set Construction.
    Handles subsampling, feature engineering, and vectorized subtraction for base margins.
    """

    def __init__(self):
        self.grs = GlobalRouteStats()

    def _process_features(self, df):
        """
        Applies common feature engineering steps.
        """
        # Clamp coordinates to NYC bounding box
        df = clamp_coordinates(df)

        # Extract temporal features
        df = extract_datetime_features(df)

        # Add distance features (Haversine, Manhattan, Rotated)
        df = add_distance_features(df)

        return df

    def _add_grid_keys(self, df):
        """
        Adds discretized coordinate keys for joining with global stats.
        """
        df["p_lat_g"] = discretize_coordinates(df["pickup_latitude"], GRID_PRECISION)
        df["p_lon_g"] = discretize_coordinates(df["pickup_longitude"], GRID_PRECISION)
        df["d_lat_g"] = discretize_coordinates(df["dropoff_latitude"], GRID_PRECISION)
        df["d_lon_g"] = discretize_coordinates(df["dropoff_longitude"], GRID_PRECISION)
        return df

    def get_train_data(self, load_cached_data=True):
        """
        Prepares the training set with Vectorized Subtraction.
        """
        cache_file = os.path.join(WORKING_DIR, "processed_train_subsample.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading processed training data from {cache_file}")
            return pd.read_parquet(cache_file)

        print("Processing training data (Stage 2)...")
        # 1. Load Global Stats
        g_stats, g_per_km = self.grs.compute_and_cache(
            load_cached_data=load_cached_data
        )

        # 2. Load Raw Train Data
        df = pd.read_parquet(TRAIN_DATA_PATH, engine="pyarrow")

        # 3. Apply Physics-Consistent Filtering (Before Subsampling)
        # Filter first to ensure the subsample contains only high-quality data.
        print("Filtering training data...")
        dist_temp = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        fare_per_km = df["fare_amount"] / (dist_temp + 1e-6)
        mask = (
            (df["fare_amount"] > 0)
            & (df["fare_amount"] <= MAX_FARE)
            & (fare_per_km <= MAX_FARE_PER_KM)
        )
        df = df.loc[mask].copy()

        # 4. Subsample
        if len(df) > SUBSAMPLE_SIZE:
            print(f"Subsampling training data to {SUBSAMPLE_SIZE} rows...")
            df = df.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_SEED).copy()
        else:
            df = df.copy()

        print(f"Final training set shape: {df.shape}")

        # 5. Feature Engineering
        df = self._process_features(df)
        df = self._add_grid_keys(df)

        # 6. Vectorized Subtraction Logic
        # Calculate Batch Stats
        batch_stats = (
            df.groupby(["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"])["fare_amount"]
            .agg(["sum", "count"])
            .reset_index()
        )
        batch_stats.rename(
            columns={"sum": "batch_sum", "count": "batch_count"}, inplace=True
        )

        # Merge Global and Batch Stats
        df = df.merge(
            g_stats, on=["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"], how="left"
        )
        df = df.merge(
            batch_stats, on=["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"], how="left"
        )

        # Fill NaNs (if route not in global stats, implied 0)
        df["global_sum"] = df["global_sum"].fillna(0)
        df["global_count"] = df["global_count"].fillna(0)

        # Calculate Residual Prior
        numerator = df["global_sum"] - df["batch_sum"]
        denominator = df["global_count"] - df["batch_count"]

        # Fallback calculation
        fallback_margin = df["distance_haversine"] * g_per_km

        # Calculate Base Margin
        # Use prior if enough data remains after subtraction, else use fallback
        # We use a safe division
        prior_margin = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

        df["base_margin"] = np.where(denominator > 0, prior_margin, fallback_margin)

        # 7. Cleanup
        drop_cols = [
            "p_lat_g",
            "p_lon_g",
            "d_lat_g",
            "d_lon_g",
            "global_sum",
            "global_count",
            "batch_sum",
            "batch_count",
        ]
        df.drop(columns=drop_cols, inplace=True)

        # 8. Save
        print(f"Saving processed training data to {cache_file}")
        df.to_parquet(cache_file)
        return df

    def get_val_data(self, load_cached_data=True):
        """
        Prepares the validation set. Uses Global Stats without subtraction.
        """
        cache_file = os.path.join(WORKING_DIR, "processed_val.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading processed validation data from {cache_file}")
            return pd.read_parquet(cache_file)

        print("Processing validation data...")
        g_stats, g_per_km = self.grs.compute_and_cache(
            load_cached_data=load_cached_data
        )

        df = pd.read_parquet(VAL_DATA_PATH, engine="pyarrow")

        # Filter validation data for consistent metric evaluation (optional but recommended for stability)
        dist_temp = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        fare_per_km = df["fare_amount"] / (dist_temp + 1e-6)
        mask = (
            (df["fare_amount"] > 0)
            & (df["fare_amount"] <= MAX_FARE)
            & (fare_per_km <= MAX_FARE_PER_KM)
        )
        df = df.loc[mask].copy()

        df = self._process_features(df)
        df = self._add_grid_keys(df)

        # Merge Global Stats
        df = df.merge(
            g_stats, on=["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"], how="left"
        )

        # Calculate Base Margin (Direct Lookup)
        prior_margin = df["global_sum"] / df["global_count"]
        fallback_margin = df["distance_haversine"] * g_per_km

        df["base_margin"] = prior_margin.fillna(fallback_margin)

        # Cleanup
        drop_cols = [
            "p_lat_g",
            "p_lon_g",
            "d_lat_g",
            "d_lon_g",
            "global_sum",
            "global_count",
        ]
        df.drop(columns=drop_cols, inplace=True)

        print(f"Saving processed validation data to {cache_file}")
        df.to_parquet(cache_file)
        return df

    def get_test_data(self, load_cached_data=True):
        """
        Prepares the test set. Uses Global Stats lookup.
        """
        cache_file = os.path.join(WORKING_DIR, "processed_test.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading processed test data from {cache_file}")
            return pd.read_parquet(cache_file)

        print("Processing test data...")
        g_stats, g_per_km = self.grs.compute_and_cache(
            load_cached_data=load_cached_data
        )

        df = pd.read_parquet(TEST_DATA_PATH, engine="pyarrow")

        # No filtering for test data
        df = self._process_features(df)
        df = self._add_grid_keys(df)

        # Merge Global Stats
        df = df.merge(
            g_stats, on=["p_lat_g", "p_lon_g", "d_lat_g", "d_lon_g"], how="left"
        )

        # Calculate Base Margin
        prior_margin = df["global_sum"] / df["global_count"]
        fallback_margin = df["distance_haversine"] * g_per_km

        df["base_margin"] = prior_margin.fillna(fallback_margin)

        # Cleanup
        drop_cols = [
            "p_lat_g",
            "p_lon_g",
            "d_lat_g",
            "d_lon_g",
            "global_sum",
            "global_count",
        ]
        df.drop(columns=drop_cols, inplace=True)

        print(f"Saving processed test data to {cache_file}")
        df.to_parquet(cache_file)
        return df
