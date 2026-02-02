import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import (
    WORKING_DIR,
    BB_MIN_LAT,
    BB_MAX_LAT,
    BB_MIN_LON,
    BB_MAX_LON,
    ROTATION_ANGLE,
    LANDMARKS,
    GRID_RESOLUTION,
)
from library.utils import reduce_mem_usage


class FeatureEngineer:
    """
    Manages feature engineering for both Tree-based and Neural Network pipelines.
    Handles shared features, specific transformations, and scaler state.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_path = os.path.join(WORKING_DIR, "scaler.joblib")
        self.nn_continuous_cols = []  # To be populated during processing

    def _add_shared_features(self, df):
        """
        Adds features common to both pipelines:
        - Haversine distance
        - Rotated coordinates
        - Landmark distances
        """
        df = df.copy()

        # 1. Haversine Distance
        R = 6371.0  # Earth radius in km
        lat1 = np.radians(df["pickup_latitude"])
        lon1 = np.radians(df["pickup_longitude"])
        lat2 = np.radians(df["dropoff_latitude"])
        lon2 = np.radians(df["dropoff_longitude"])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df["haversine_dist"] = R * c

        # 2. Coordinate Rotation (NYC Grid Alignment)
        # x' = x cos(theta) - y sin(theta)
        # y' = x sin(theta) + y cos(theta)
        theta = np.radians(ROTATION_ANGLE)
        c, s = np.cos(theta), np.sin(theta)

        # Rotate Pickup
        df["pickup_rot_x"] = df["pickup_longitude"] * c - df["pickup_latitude"] * s
        df["pickup_rot_y"] = df["pickup_longitude"] * s + df["pickup_latitude"] * c

        # Rotate Dropoff
        df["dropoff_rot_x"] = df["dropoff_longitude"] * c - df["dropoff_latitude"] * s
        df["dropoff_rot_y"] = df["dropoff_longitude"] * s + df["dropoff_latitude"] * c

        # 3. Landmark Distances
        for name, (lat, lon) in LANDMARKS.items():
            # Simplified Euclidean distance on lat/lon for speed (sufficient for features)
            # or Haversine. Using simple Euclidean on coords is often enough for trees,
            # but Haversine is more accurate. Let's use a vectorized approximation.
            # d = sqrt((lat1-lat2)^2 + (lon1-lon2)^2)
            # Note: This is "degree distance", not km, but valid for ML.
            df[f"dist_to_{name}_pickup"] = np.sqrt(
                (df["pickup_latitude"] - lat) ** 2 + (df["pickup_longitude"] - lon) ** 2
            )
            df[f"dist_to_{name}_dropoff"] = np.sqrt(
                (df["dropoff_latitude"] - lat) ** 2
                + (df["dropoff_longitude"] - lon) ** 2
            )

        # 4. Directional features
        df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]
        df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]

        return df

    def _get_time_features(self, df):
        """Extracts basic time components."""
        dt = pd.to_datetime(df["pickup_datetime"], utc=True)
        return pd.DataFrame(
            {
                "hour": dt.dt.hour,
                "day": dt.dt.day,
                "month": dt.dt.month,
                "year": dt.dt.year,
                "dayofweek": dt.dt.dayofweek,
            },
            index=df.index,
        )

    def fit_scaler(self, df):
        """
        Fits the StandardScaler on the continuous columns required for the NN pipeline.
        Must be called on the training set.
        """
        # Generate shared features first to identify columns
        df_shared = self._add_shared_features(df)

        # Identify continuous columns for scaling
        # We exclude ID, target, and raw time components (which will be cyclic or embedded)
        exclude_cols = [
            "key",
            "pickup_datetime",
            "fare_amount",
            "pickup_grid_x",
            "pickup_grid_y",
            "dropoff_grid_x",
            "dropoff_grid_y",
        ]

        # We want to scale coordinates, distances, rotated coords
        # Basically all float columns generated + original coords + passenger_count
        candidates = [
            c
            for c in df_shared.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_shared[c])
        ]

        self.nn_continuous_cols = candidates

        print(f"Fitting scaler on {len(candidates)} columns: {candidates}")
        self.scaler.fit(df_shared[candidates])

        # Save scaler state
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.nn_continuous_cols, self.scaler_path + ".cols")

    def load_scaler(self):
        """Loads the fitted scaler."""
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            self.nn_continuous_cols = joblib.load(self.scaler_path + ".cols")
        else:
            raise FileNotFoundError("Scaler not found. Call fit_scaler first.")

    def transform_tree(self, df):
        """
        Pipeline for Tree-based models (XGBoost, LightGBM).
        - Raw integer time features.
        - Raw coordinates.
        - Shared engineered features.
        """
        # 1. Shared Features
        df_eng = self._add_shared_features(df)

        # 2. Time Features (Raw Integers)
        time_df = self._get_time_features(df)
        df_eng = pd.concat([df_eng, time_df], axis=1)

        # 3. Cleanup
        # Drop non-numeric columns not needed for trees
        drop_cols = ["key", "pickup_datetime"]
        df_eng = df_eng.drop(columns=[c for c in drop_cols if c in df_eng.columns])

        return reduce_mem_usage(df_eng)

    def transform_nn(self, df):
        """
        Pipeline for Neural Network (Spatial ResNet).
        - Scaled continuous features.
        - Cyclical time features.
        - Grid Index Embeddings for coordinates.
        """
        # Ensure scaler is loaded
        if not hasattr(self.scaler, "mean_"):
            self.load_scaler()

        # 1. Shared Features
        df_eng = self._add_shared_features(df)

        # 2. Grid Embeddings (Discretization)
        # Calculate indices based on resolution relative to bounding box min
        # Clamp to ensure we don't go out of bounds (though cleaning should handle this)
        # We add a small buffer to max bins calculation to be safe

        # Lon Range: ~2.0 deg. Res: 0.002. ~1000 bins.
        def get_grid_idx(val, min_val, max_val, res):
            idx = ((val - min_val) / res).astype(int)
            max_idx = int((max_val - min_val) / res)
            return np.clip(idx, 0, max_idx)

        df_eng["grid_pickup_lat"] = get_grid_idx(
            df["pickup_latitude"], BB_MIN_LAT, BB_MAX_LAT, GRID_RESOLUTION
        )
        df_eng["grid_pickup_lon"] = get_grid_idx(
            df["pickup_longitude"], BB_MIN_LON, BB_MAX_LON, GRID_RESOLUTION
        )
        df_eng["grid_dropoff_lat"] = get_grid_idx(
            df["dropoff_latitude"], BB_MIN_LAT, BB_MAX_LAT, GRID_RESOLUTION
        )
        df_eng["grid_dropoff_lon"] = get_grid_idx(
            df["dropoff_longitude"], BB_MIN_LON, BB_MAX_LON, GRID_RESOLUTION
        )

        # 3. Time Features (Cyclical)
        time_df = self._get_time_features(df)

        # Hour (24)
        df_eng["hour_sin"] = np.sin(2 * np.pi * time_df["hour"] / 24)
        df_eng["hour_cos"] = np.cos(2 * np.pi * time_df["hour"] / 24)

        # Day of Week (7)
        df_eng["day_sin"] = np.sin(2 * np.pi * time_df["dayofweek"] / 7)
        df_eng["day_cos"] = np.cos(2 * np.pi * time_df["dayofweek"] / 7)

        # Month (12)
        df_eng["month_sin"] = np.sin(2 * np.pi * time_df["month"] / 12)
        df_eng["month_cos"] = np.cos(2 * np.pi * time_df["month"] / 12)

        # 4. Scaling Continuous Features
        # Apply standard scaler to the columns identified during fit
        # Note: We must handle the case where df_eng might have fewer columns if some were dropped,
        # but here we are building it up.

        # We need to ensure the columns exist.
        cols_to_scale = [c for c in self.nn_continuous_cols if c in df_eng.columns]
        df_eng[cols_to_scale] = self.scaler.transform(df_eng[cols_to_scale])

        # 5. Cleanup
        # Keep: Scaled Continuous, Grid Indices, Cyclical Time, Target (if exists), Key (if test)
        # Drop: Raw Time, Raw Coords (if they are in scaled list, they are already transformed in place),
        # Original String columns

        keep_cols = (
            cols_to_scale
            + [
                "grid_pickup_lat",
                "grid_pickup_lon",
                "grid_dropoff_lat",
                "grid_dropoff_lon",
            ]
            + ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
        )

        if "fare_amount" in df.columns:
            keep_cols.append("fare_amount")
        if "key" in df.columns:
            keep_cols.append("key")

        df_final = df_eng[keep_cols].copy()

        return reduce_mem_usage(df_final)


def process_features(
    df_train_base, df_train_meta, df_val, df_test, load_cached_data=True
):
    """
    Orchestrates the feature engineering process.
    Checks for cached parquet files. If not found, computes features and caches them.

    Returns:
        Dictionary containing transformed dataframes:
        {
            'train_base_tree': ..., 'train_base_nn': ...,
            'train_meta_tree': ..., 'train_meta_nn': ...,
            'val_tree': ...,        'val_nn': ...,
            'test_tree': ...,       'test_nn': ...
        }
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define file names
    files = {
        "train_base_tree": "train_base_tree.parquet",
        "train_base_nn": "train_base_nn.parquet",
        "train_meta_tree": "train_meta_tree.parquet",
        "train_meta_nn": "train_meta_nn.parquet",
        "val_tree": "val_tree.parquet",
        "val_nn": "val_nn.parquet",
        "test_tree": "test_tree.parquet",
        "test_nn": "test_nn.parquet",
    }

    # Check cache
    all_exist = all(
        os.path.exists(os.path.join(WORKING_DIR, f)) for f in files.values()
    )

    if load_cached_data and all_exist:
        print("Loading feature-engineered data from cache...")
        results = {}
        for key, filename in files.items():
            results[key] = pd.read_parquet(os.path.join(WORKING_DIR, filename))
        return results

    print("Generating features from scratch...")

    # Initialize Engineer
    engineer = FeatureEngineer()

    # 1. Fit Scaler (on Base Train only)
    print("Fitting scalers on Base Train...")
    engineer.fit_scaler(df_train_base)

    results = {}

    # Helper to process and save
    def process_and_save(df, prefix):
        print(f"Transforming {prefix} (Tree)...")
        df_tree = engineer.transform_tree(df)
        path_tree = os.path.join(WORKING_DIR, files[f"{prefix}_tree"])
        df_tree.to_parquet(path_tree, index=False)
        results[f"{prefix}_tree"] = df_tree

        print(f"Transforming {prefix} (NN)...")
        df_nn = engineer.transform_nn(df)
        path_nn = os.path.join(WORKING_DIR, files[f"{prefix}_nn"])
        df_nn.to_parquet(path_nn, index=False)
        results[f"{prefix}_nn"] = df_nn

    # 2. Transform Datasets
    process_and_save(df_train_base, "train_base")
    process_and_save(df_train_meta, "train_meta")
    process_and_save(df_val, "val")
    process_and_save(df_test, "test")

    print("Feature engineering complete.")
    return results
