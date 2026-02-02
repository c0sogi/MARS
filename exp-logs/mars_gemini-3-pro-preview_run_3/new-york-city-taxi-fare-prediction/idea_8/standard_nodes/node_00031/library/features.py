import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library import config, utils


class FeatureFactory:
    """
    Encapsulates the feature engineering logic for the Dual-Pipeline strategy.
    Handles both Tree-based (raw/integer) and Neural Network (scaled/cyclical) transformations.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _haversine_np(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees) using NumPy.
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371.0  # Radius of earth in kilometers
        return c * r

    def _add_shared_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features common to both pipelines:
        - Distance metrics (Haversine, Manhattan, Euclidean)
        - Rotated coordinates
        - Landmark distances
        - Basic time extraction (needed for downstream specific processing)
        """
        df = df.copy()

        # Ensure datetime
        if "pickup_datetime" in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
                df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

            # Extract basic time components
            dt = df["pickup_datetime"].dt
            df["hour"] = dt.hour
            df["day"] = dt.day
            df["month"] = dt.month
            df["year"] = dt.year
            df["day_of_week"] = dt.dayofweek

        # Coordinates
        p_lon, p_lat = df["pickup_longitude"], df["pickup_latitude"]
        d_lon, d_lat = df["dropoff_longitude"], df["dropoff_latitude"]

        # 1. Distance Metrics
        df["abs_diff_longitude"] = (d_lon - p_lon).abs()
        df["abs_diff_latitude"] = (d_lat - p_lat).abs()
        df["manhattan_dist"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]
        df["euclidean_dist"] = np.sqrt(
            df["abs_diff_longitude"] ** 2 + df["abs_diff_latitude"] ** 2
        )
        df["haversine_dist"] = self._haversine_np(p_lon, p_lat, d_lon, d_lat)

        # 2. Rotated Coordinates (NYC Grid Alignment)
        # Rotation formula: x' = x cos(theta) - y sin(theta), y' = x sin(theta) + y cos(theta)
        theta = np.radians(config.ROTATION_ANGLE)
        c, s = np.cos(theta), np.sin(theta)

        df["pickup_rot_x"] = p_lon * c - p_lat * s
        df["pickup_rot_y"] = p_lon * s + p_lat * c
        df["dropoff_rot_x"] = d_lon * c - d_lat * s
        df["dropoff_rot_y"] = d_lon * s + d_lat * c

        # 3. Landmark Distances
        for name, (lat, lon) in config.LANDMARKS.items():
            # Calculate distance from Pickup to Landmark
            # (Could also do Dropoff, but Pickup is usually more indicative of fare structure logic like airport fees)
            df[f"dist_to_{name}"] = self._haversine_np(
                p_lon,
                p_lat,
                pd.Series(lon, index=df.index),
                pd.Series(lat, index=df.index),
            )

        return df

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the internal StandardScaler on the continuous features of the training set
        required for the Neural Network pipeline.
        """
        print("Fitting FeatureFactory scaler...")
        # Generate the features first
        processed = self._add_shared_features(train_df)

        # Select columns to scale
        cols_to_fit = config.NN_CONTINUOUS_FEATURES

        # Verify columns exist
        missing = [c for c in cols_to_fit if c not in processed.columns]
        if missing:
            raise ValueError(f"Missing columns for scaling: {missing}")

        self.scaler.fit(processed[cols_to_fit])
        self.is_fitted = True
        print("Scaler fitted successfully.")

    def transform_tree(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data for Tree-based models (XGBoost/LightGBM).
        - Uses raw integers for time.
        - Uses unscaled coordinates and distances.
        """
        # Generate shared features
        df_eng = self._add_shared_features(df)

        # Select specific features for Trees
        # (Base + Shared + Tree Time)
        cols = config.TREE_FEATURES

        # Add target if present
        if "fare_amount" in df.columns:
            return df_eng[cols + ["fare_amount"]]
        else:
            # Keep key for test set submission mapping if needed,
            # though usually we just predict by row order.
            # Returning just features for model input.
            return df_eng[cols]

    def transform_nn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data for the Neural Network (ResNet).
        - StandardScales continuous features.
        - Applies Sin/Cos transformation to cyclic time features.
        - Preserves categorical integers for embeddings.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureFactory must be fitted before calling transform_nn"
            )

        # Generate shared features
        df_eng = self._add_shared_features(df)

        # 1. Scale Continuous Features
        cont_cols = config.NN_CONTINUOUS_FEATURES
        scaled_data = self.scaler.transform(df_eng[cont_cols])
        df_scaled = pd.DataFrame(scaled_data, columns=cont_cols, index=df_eng.index)

        # 2. Cyclical Features (Sin/Cos)
        # Hour (0-23)
        df_scaled["hour_sin"] = np.sin(2 * np.pi * df_eng["hour"] / 24.0)
        df_scaled["hour_cos"] = np.cos(2 * np.pi * df_eng["hour"] / 24.0)

        # Month (1-12)
        df_scaled["month_sin"] = np.sin(2 * np.pi * df_eng["month"] / 12.0)
        df_scaled["month_cos"] = np.cos(2 * np.pi * df_eng["month"] / 12.0)

        # 3. Categorical Features (for Embeddings)
        # Passenger count (clip to prevent index errors if > max expected)
        # Config says embedding dim 10, so values 0-9.
        df_scaled["passenger_count"] = df_eng["passenger_count"].clip(0, 9).astype(int)

        # Day of week (0-6)
        df_scaled["day_of_week"] = df_eng["day_of_week"].astype(int)

        # Add target if present
        if "fare_amount" in df.columns:
            df_scaled["fare_amount"] = df_eng["fare_amount"]

        return df_scaled

    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        self.scaler = joblib.load(path)
        self.is_fitted = True


def process_data(train_df, val_df, test_df, load_cached_data=True):
    """
    Orchestrates the feature engineering process with caching.

    Args:
        train_df, val_df, test_df: Raw cleaned dataframes.
        load_cached_data: Boolean to determine if cache should be used.

    Returns:
        dict: Contains 'train_tree', 'val_tree', 'test_tree', 'train_nn', 'val_nn', 'test_nn' dataframes.
    """
    utils.seed_everything(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Check if all cache files exist
    cache_files = [
        config.CACHE_TRAIN_TREE,
        config.CACHE_VAL_TREE,
        config.CACHE_TEST_TREE,
        config.CACHE_TRAIN_NN,
        config.CACHE_VAL_NN,
        config.CACHE_TEST_NN,
        config.CACHE_SCALER,
    ]

    all_exist = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and all_exist:
        print("Loading engineered features from cache...")
        data = {
            "train_tree": pd.read_parquet(config.CACHE_TRAIN_TREE),
            "val_tree": pd.read_parquet(config.CACHE_VAL_TREE),
            "test_tree": pd.read_parquet(config.CACHE_TEST_TREE),
            "train_nn": pd.read_parquet(config.CACHE_TRAIN_NN),
            "val_nn": pd.read_parquet(config.CACHE_VAL_NN),
            "test_nn": pd.read_parquet(config.CACHE_TEST_NN),
        }
        # Load scaler just in case it's needed later, though not returned in dict
        # factory = FeatureFactory()
        # factory.load_scaler(config.CACHE_SCALER)
        return data

    print("Processing features from scratch...")

    # Initialize Factory
    factory = FeatureFactory()

    # Fit on Training Data
    factory.fit(train_df)
    factory.save_scaler(config.CACHE_SCALER)

    # Transform Tree Data
    print("Generating Tree features...")
    train_tree = factory.transform_tree(train_df)
    val_tree = factory.transform_tree(val_df)
    test_tree = factory.transform_tree(test_df)

    # Transform NN Data
    print("Generating Neural Network features...")
    train_nn = factory.transform_nn(train_df)
    val_nn = factory.transform_nn(val_df)
    test_nn = factory.transform_nn(test_df)

    # Save to Cache
    print("Saving to cache...")
    train_tree.to_parquet(config.CACHE_TRAIN_TREE, index=False)
    val_tree.to_parquet(config.CACHE_VAL_TREE, index=False)
    test_tree.to_parquet(config.CACHE_TEST_TREE, index=False)

    train_nn.to_parquet(config.CACHE_TRAIN_NN, index=False)
    val_nn.to_parquet(config.CACHE_VAL_NN, index=False)
    test_nn.to_parquet(config.CACHE_TEST_NN, index=False)

    return {
        "train_tree": train_tree,
        "val_tree": val_tree,
        "test_tree": test_tree,
        "train_nn": train_nn,
        "val_nn": val_nn,
        "test_nn": test_nn,
    }
