import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import haversine_distance, manhattan_distance


class DataProcessor:
    def __init__(self):
        # Continuous features to be standardized
        self.continuous_cols = [
            "passenger_count",
            "haversine_dist",
            "manhattan_dist",
            "pickup_rot_1",
            "pickup_rot_2",
            "dropoff_rot_1",
            "dropoff_rot_2",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]

        # Categorical features for embeddings (indices)
        self.categorical_cols = [
            "pickup_grid_lat",
            "pickup_grid_lon",
            "dropoff_grid_lat",
            "dropoff_grid_lon",
            "hour",
            "weekday",
            "day",
            "month",
            "year",
        ]

        self.scaler_mean = None
        self.scaler_scale = None

    def _get_paths(self):
        """Returns a dictionary of file paths for cached data."""
        return {
            "train": os.path.join(Config.WORKING_DIR, "train_processed.parquet"),
            "val": os.path.join(Config.WORKING_DIR, "val_processed.parquet"),
            "test": os.path.join(Config.WORKING_DIR, "test_processed.parquet"),
            "scaler_mean": os.path.join(Config.WORKING_DIR, "scaler_mean.npy"),
            "scaler_scale": os.path.join(Config.WORKING_DIR, "scaler_scale.npy"),
        }

    def clean_data(self, df):
        """
        Clamps spatial coordinates to the bounding box defined in Config.
        This prevents outliers from distorting the model.
        """
        bounds = Config.get_spatial_bounds()

        df["pickup_latitude"] = df["pickup_latitude"].clip(
            bounds["lat_min"], bounds["lat_max"]
        )
        df["pickup_longitude"] = df["pickup_longitude"].clip(
            bounds["lon_min"], bounds["lon_max"]
        )

        df["dropoff_latitude"] = df["dropoff_latitude"].clip(
            bounds["lat_min"], bounds["lat_max"]
        )
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(
            bounds["lon_min"], bounds["lon_max"]
        )

        return df

    def compute_grid_indices(self, series, min_val, max_val, bins):
        """
        Maps continuous values to integer grid indices [0, bins-1].
        Used for spatial embeddings.
        """
        # Normalize to [0, 1]
        normalized = (series - min_val) / (max_val - min_val)
        # Scale to bins
        indices = (normalized * bins).astype(int)
        # Clip to ensure bounds
        return indices.clip(0, bins - 1)

    def add_features(self, df):
        """
        Generates spatial and temporal features.
        """
        # 1. Temporal Features
        # Ensure datetime is datetime object
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Handle potential " UTC" suffix if parsing fails or just strip it for safety
            if df["pickup_datetime"].dtype == "object":
                df["pickup_datetime"] = (
                    df["pickup_datetime"].astype(str).str.replace(" UTC", "")
                )
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        df["hour"] = df["pickup_datetime"].dt.hour
        df["weekday"] = df["pickup_datetime"].dt.dayofweek
        df["day"] = df["pickup_datetime"].dt.day
        df["month"] = df["pickup_datetime"].dt.month
        # Normalize year to 0-based index (2009 is base)
        df["year"] = df["pickup_datetime"].dt.year - 2009

        # 2. Distance Features
        df["haversine_dist"] = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )
        df["manhattan_dist"] = manhattan_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # 3. Rotated Coordinates (helps with diagonal streets)
        # Rot 1: Lat + Lon
        df["pickup_rot_1"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["dropoff_rot_1"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        # Rot 2: Lat - Lon
        df["pickup_rot_2"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_rot_2"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        # 4. Grid Indices (Spatial Embeddings)
        bounds = Config.get_spatial_bounds()
        df["pickup_grid_lat"] = self.compute_grid_indices(
            df["pickup_latitude"],
            bounds["lat_min"],
            bounds["lat_max"],
            Config.GRID_BINS,
        )
        df["pickup_grid_lon"] = self.compute_grid_indices(
            df["pickup_longitude"],
            bounds["lon_min"],
            bounds["lon_max"],
            Config.GRID_BINS,
        )
        df["dropoff_grid_lat"] = self.compute_grid_indices(
            df["dropoff_latitude"],
            bounds["lat_min"],
            bounds["lat_max"],
            Config.GRID_BINS,
        )
        df["dropoff_grid_lon"] = self.compute_grid_indices(
            df["dropoff_longitude"],
            bounds["lon_min"],
            bounds["lon_max"],
            Config.GRID_BINS,
        )

        return df

    def fit_scaler(self, df):
        """
        Computes mean and std for continuous columns from the training set.
        """
        means = []
        scales = []
        for col in self.continuous_cols:
            vals = df[col].values
            mu = np.mean(vals)
            sigma = np.std(vals)
            if sigma == 0:
                sigma = 1.0  # Avoid division by zero
            means.append(mu)
            scales.append(sigma)

        self.scaler_mean = np.array(means, dtype=np.float32)
        self.scaler_scale = np.array(scales, dtype=np.float32)

    def transform_scaler(self, df):
        """
        Applies standardization using stored mean and scale.
        """
        if self.scaler_mean is None or self.scaler_scale is None:
            raise ValueError("Scaler has not been fitted.")

        for i, col in enumerate(self.continuous_cols):
            # Apply (x - mu) / sigma
            # Use .values to avoid index alignment issues and speed up
            df[col] = (df[col].values - self.scaler_mean[i]) / self.scaler_scale[i]
            # Cast to float32 for memory efficiency
            df[col] = df[col].astype(np.float32)

        return df

    def process_data(self, load_cached_data=True):
        """
        Main entry point for data processing.
        Loads raw data, cleans, generates features, scales, and caches results.
        """
        paths = self._get_paths()
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Try to load cached data
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in paths.values())
            if all_exist:
                print("Loading cached processed data...")
                try:
                    train_df = pd.read_parquet(paths["train"])
                    val_df = pd.read_parquet(paths["val"])
                    test_df = pd.read_parquet(paths["test"])

                    # Load scaler params
                    self.scaler_mean = np.load(paths["scaler_mean"])
                    self.scaler_scale = np.load(paths["scaler_scale"])

                    return train_df, val_df, test_df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing...")
            else:
                print("Cache incomplete. Reprocessing...")

        # 2. Process from scratch
        print("Processing data from scratch...")

        # Load Raw Data
        print("Loading raw parquet files...")
        train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)
        test_df = pd.read_parquet(Config.TEST_DATA_PATH)

        # Debugging: Subsample if configured
        if Config.DEBUG:
            print(f"DEBUG Mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows.")
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Clean Data
        print("Cleaning data (clamping coordinates)...")
        train_df = self.clean_data(train_df)
        val_df = self.clean_data(val_df)
        test_df = self.clean_data(test_df)

        # Feature Engineering
        print("Generating features...")
        train_df = self.add_features(train_df)
        val_df = self.add_features(val_df)
        test_df = self.add_features(test_df)

        # Fit Scaler (only on Train)
        print("Fitting scaler on training data...")
        self.fit_scaler(train_df)

        # Transform Scaler
        print("Applying scaler to all splits...")
        train_df = self.transform_scaler(train_df)
        val_df = self.transform_scaler(val_df)
        test_df = self.transform_scaler(test_df)

        # Save to Cache
        print("Saving processed data to cache...")
        train_df.to_parquet(paths["train"], index=False)
        val_df.to_parquet(paths["val"], index=False)
        test_df.to_parquet(paths["test"], index=False)

        np.save(paths["scaler_mean"], self.scaler_mean)
        np.save(paths["scaler_scale"], self.scaler_scale)

        return train_df, val_df, test_df
