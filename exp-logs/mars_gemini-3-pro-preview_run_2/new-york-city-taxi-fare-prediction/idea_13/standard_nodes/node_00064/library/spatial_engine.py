import os
import numpy as np
import pandas as pd
import gc
from library.config import (
    WORKING_DIR,
    SPATIAL_RESOLUTIONS,
    NUM_FOLDS,
    NYC_BOUNDING_BOX,
    RANDOM_SEED,
)
from library.utils import haversine_distance, clamp_values


class SpatialEngine:
    """
    Implements the Multi-Resolution Global-Prior logic.
    Generates spatial features using K-Fold Vectorized Subtraction.
    """

    def __init__(self):
        self.cache_dir = os.path.join(WORKING_DIR, "spatial_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.global_mean = None

    def _clamp_coordinates(self, df):
        """Clamps pickup and dropoff coordinates to the NYC bounding box."""
        df["pickup_latitude"] = clamp_values(
            df["pickup_latitude"],
            NYC_BOUNDING_BOX["min_lat"],
            NYC_BOUNDING_BOX["max_lat"],
        )
        df["pickup_longitude"] = clamp_values(
            df["pickup_longitude"],
            NYC_BOUNDING_BOX["min_lon"],
            NYC_BOUNDING_BOX["max_lon"],
        )
        df["dropoff_latitude"] = clamp_values(
            df["dropoff_latitude"],
            NYC_BOUNDING_BOX["min_lat"],
            NYC_BOUNDING_BOX["max_lat"],
        )
        df["dropoff_longitude"] = clamp_values(
            df["dropoff_longitude"],
            NYC_BOUNDING_BOX["min_lon"],
            NYC_BOUNDING_BOX["max_lon"],
        )
        return df

    def _get_clean_mask(self, df):
        """
        Generates a boolean mask for 'clean' data to be used for statistics generation.
        Criteria:
        1. Fare >= 2.5 (Minimum fare)
        2. Fare <= 500 (Remove extreme outliers)
        3. Fare / Distance <= 50 per km (Remove unrealistic price/distance ratios)
        """
        # Calculate distance for hygiene check
        dist = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Avoid division by zero
        dist = np.maximum(dist, 0.001)

        # Criteria
        mask = (
            (df["fare_amount"] >= 2.5)
            & (df["fare_amount"] <= 500.0)
            & ((df["fare_amount"] / dist) <= 50.0)
        )
        return mask

    def compute_stats_at_resolution(self, df, resolution, target_col="fare_amount"):
        """
        Aggregates sum and count of target variable at a specific spatial resolution.
        Returns dictionaries for pickup and dropoff stats.
        """
        # Create rounded coordinates
        p_lat = df["pickup_latitude"].round(resolution)
        p_lon = df["pickup_longitude"].round(resolution)
        d_lat = df["dropoff_latitude"].round(resolution)
        d_lon = df["dropoff_longitude"].round(resolution)

        # Groupby Pickup
        pickup_stats = df.groupby([p_lat, p_lon])[target_col].agg(["sum", "count"])
        pickup_stats.columns = ["sum", "count"]

        # Groupby Dropoff
        dropoff_stats = df.groupby([d_lat, d_lon])[target_col].agg(["sum", "count"])
        dropoff_stats.columns = ["sum", "count"]

        return pickup_stats, dropoff_stats

    def generate_kfold_priors(self, df, load_cached_data=True):
        """
        Generates Multi-Resolution Priors for the training set using Vectorized Subtraction.
        Uses Route-Based (Interaction) Target Encoding. Cite solution_lesson_node_00062.

        Args:
            df (pd.DataFrame): The full training dataset.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The dataframe with added feature columns.
        """
        cache_path = os.path.join(self.cache_dir, "train_with_priors.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached spatial priors from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Generating K-Fold Spatial Priors (Vectorized Subtraction)...")

        # 1. Hygiene & Setup
        df = self._clamp_coordinates(df.copy())
        clean_mask = self._get_clean_mask(df)

        # Calculate global mean from clean data for fillna
        self.global_mean = df.loc[clean_mask, "fare_amount"].mean()
        print(f"Global Clean Mean Fare: {self.global_mean:.4f}")

        # Assign Folds deterministically
        rng = np.random.RandomState(RANDOM_SEED)
        df["fold"] = rng.randint(0, NUM_FOLDS, size=len(df))

        # We process clean data for stats
        clean_df = df[clean_mask].copy()

        # 2. Iterate Resolutions
        for res in SPATIAL_RESOLUTIONS:
            print(f"Processing Resolution: {res} decimal places...")

            # --- PREPARE COORDINATES ---
            p_lat_col = f"p_lat_r{res}"
            p_lon_col = f"p_lon_r{res}"
            d_lat_col = f"d_lat_r{res}"
            d_lon_col = f"d_lon_r{res}"

            route_cols = [p_lat_col, p_lon_col, d_lat_col, d_lon_col]

            for d in [df, clean_df]:
                d[p_lat_col] = d["pickup_latitude"].round(res)
                d[p_lon_col] = d["pickup_longitude"].round(res)
                d[d_lat_col] = d["dropoff_latitude"].round(res)
                d[d_lon_col] = d["dropoff_longitude"].round(res)

            # --- ROUTE PRIORS (Interaction) ---
            # Global Stats (on clean data)
            g_stats = (
                clean_df.groupby(route_cols)["fare_amount"]
                .agg(["sum", "count"])
                .reset_index()
            )
            g_stats.rename(columns={"sum": "g_sum", "count": "g_count"}, inplace=True)

            # Fold Stats (on clean data)
            f_stats = (
                clean_df.groupby(["fold"] + route_cols)["fare_amount"]
                .agg(["sum", "count"])
                .reset_index()
            )
            f_stats.rename(columns={"sum": "f_sum", "count": "f_count"}, inplace=True)

            # Merge Global Stats to Full DF
            df = df.merge(g_stats, on=route_cols, how="left")
            # Merge Fold Stats to Full DF
            df = df.merge(f_stats, on=["fold"] + route_cols, how="left")

            # Fill NaNs (No stats found)
            df["g_sum"] = df["g_sum"].fillna(0)
            df["g_count"] = df["g_count"].fillna(0)
            df["f_sum"] = df["f_sum"].fillna(0)
            df["f_count"] = df["f_count"].fillna(0)

            # Vectorized Subtraction
            numerator = df["g_sum"] - df["f_sum"]
            denominator = df["g_count"] - df["f_count"]

            # Calculate Mean, handle division by zero
            # If denominator is 0 (no data in other folds), use global mean
            df[f"route_mean_{res}"] = np.where(
                denominator > 0, numerator / denominator, self.global_mean
            ).astype(np.float32)

            # Cleanup temp columns
            df.drop(
                columns=["g_sum", "g_count", "f_sum", "f_count"] + route_cols,
                inplace=True,
            )

            gc.collect()

        # Remove fold column
        df.drop(columns=["fold"], inplace=True)

        # Cache result
        print(f"Saving cached spatial priors to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df

    def apply_global_priors(self, test_df, train_df):
        """
        Applies Global Priors (computed from train_df) to the test set.
        Uses Route-Based (Interaction) Target Encoding.

        Args:
            test_df (pd.DataFrame): The test dataset.
            train_df (pd.DataFrame): The training dataset (source of stats).

        Returns:
            pd.DataFrame: Test dataframe with added feature columns.
        """
        print("Applying Global Spatial Priors to Test Set...")

        # 1. Hygiene on Source
        train_df = self._clamp_coordinates(train_df.copy())
        clean_mask = self._get_clean_mask(train_df)
        clean_train = train_df[clean_mask]

        if self.global_mean is None:
            self.global_mean = clean_train["fare_amount"].mean()

        test_df = self._clamp_coordinates(test_df.copy())

        for res in SPATIAL_RESOLUTIONS:
            # --- ROUTE ---
            p_lat_col = f"p_lat_r{res}"
            p_lon_col = f"p_lon_r{res}"
            d_lat_col = f"d_lat_r{res}"
            d_lon_col = f"d_lon_r{res}"

            route_cols = [p_lat_col, p_lon_col, d_lat_col, d_lon_col]

            # Compute Global Stats
            clean_train[p_lat_col] = clean_train["pickup_latitude"].round(res)
            clean_train[p_lon_col] = clean_train["pickup_longitude"].round(res)
            clean_train[d_lat_col] = clean_train["dropoff_latitude"].round(res)
            clean_train[d_lon_col] = clean_train["dropoff_longitude"].round(res)

            g_stats = (
                clean_train.groupby(route_cols)["fare_amount"].mean().reset_index()
            )
            g_stats.rename(columns={"fare_amount": f"route_mean_{res}"}, inplace=True)

            # Prepare Test
            test_df[p_lat_col] = test_df["pickup_latitude"].round(res)
            test_df[p_lon_col] = test_df["pickup_longitude"].round(res)
            test_df[d_lat_col] = test_df["dropoff_latitude"].round(res)
            test_df[d_lon_col] = test_df["dropoff_longitude"].round(res)

            # Merge
            test_df = test_df.merge(g_stats, on=route_cols, how="left")
            test_df[f"route_mean_{res}"] = (
                test_df[f"route_mean_{res}"].fillna(self.global_mean).astype(np.float32)
            )

            # Cleanup
            test_df.drop(columns=route_cols, inplace=True)
            clean_train.drop(columns=route_cols, inplace=True)

            gc.collect()

        return test_df
