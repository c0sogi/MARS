import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    SUBSAMPLE_SIZE,
    SEED,
    CACHE_DIR,
    BBOX,
    MIN_FARE,
    MAX_FARE,
    MAX_PASSENGERS,
)
from library.utils import filter_within_bbox
from library.global_features import generate_global_features, SpatialTargetEncoder
from library.local_features import generate_local_features


class DataManager:
    def __init__(self):
        self.train_path = TRAIN_DATA_PATH
        self.val_path = VAL_DATA_PATH
        self.test_path = TEST_DATA_PATH

    def _clamp_coordinates(self, df, bbox):
        """
        Clamps coordinates to the bounding box. Used for the test set where
        rows cannot be dropped.
        """
        df = df.copy()
        df["pickup_longitude"] = df["pickup_longitude"].clip(
            bbox["min_long"], bbox["max_long"]
        )
        df["pickup_latitude"] = df["pickup_latitude"].clip(
            bbox["min_lat"], bbox["max_lat"]
        )
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(
            bbox["min_long"], bbox["max_long"]
        )
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(
            bbox["min_lat"], bbox["max_lat"]
        )
        return df

    def _get_validation_global_features(self, train_df, val_df, load_cached_data):
        """
        Generates global features for the validation set by fitting on the
        training set and transforming the validation set. Handles caching.
        """
        cache_path = os.path.join(CACHE_DIR, "val_route_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading cached validation global features...")
            try:
                val_feats = pd.read_parquet(cache_path)
                # Simple validation check
                if len(val_feats) == len(val_df):
                    return val_feats
                print("Cache size mismatch for validation features. Recomputing...")
            except Exception as e:
                print(f"Error loading validation cache: {e}. Recomputing...")

        print("Computing validation global features...")
        # Fit on train, transform val
        encoder = SpatialTargetEncoder()
        encoder.fit(train_df)
        val_route_vals = encoder.transform(val_df)

        val_feats = pd.DataFrame(
            {"key": val_df["key"].values, "route_avg_fare": val_route_vals}
        )

        # Cache result
        os.makedirs(CACHE_DIR, exist_ok=True)
        val_feats.to_parquet(cache_path, index=False)

        return val_feats

    def load_and_prepare_data(self, load_cached_data=True):
        """
        Orchestrates the data loading, cleaning, feature generation, and splitting.
        """
        print("Loading raw data...")
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)
        test_df = pd.read_parquet(self.test_path)

        # 1. Clean Data
        # Filter Train and Val (Drop outliers)
        print("Filtering training and validation data (BBox)...")
        train_df = filter_within_bbox(train_df, BBOX)
        val_df = filter_within_bbox(val_df, BBOX)

        # Filter Fare Amount and Passenger Count (Cite solution_lesson_node_00017, solution_lesson_node_00018)
        # Removing extreme outliers from target variable to stabilize RMSE loss.
        print(
            f"Filtering training and validation data (Fare: {MIN_FARE}-{MAX_FARE}, Pass: 0-{MAX_PASSENGERS})..."
        )
        train_df = train_df[
            (train_df["fare_amount"] >= MIN_FARE)
            & (train_df["fare_amount"] <= MAX_FARE)
            & (train_df["passenger_count"] > 0)
            & (train_df["passenger_count"] <= MAX_PASSENGERS)
        ]
        val_df = val_df[
            (val_df["fare_amount"] >= MIN_FARE)
            & (val_df["fare_amount"] <= MAX_FARE)
            & (val_df["passenger_count"] > 0)
            & (val_df["passenger_count"] <= MAX_PASSENGERS)
        ]

        # Clamp Test (Keep all rows, but fix outliers)
        print("Clamping test data (BBox)...")
        test_df = self._clamp_coordinates(test_df, BBOX)

        # 2. Global Features (Route Avg Fare)
        # Note: generate_global_features handles caching internally
        print("Generating global features...")
        train_global, test_global = generate_global_features(
            train_df, test_df, load_cached_data=load_cached_data
        )

        # Generate for Validation set
        val_global = self._get_validation_global_features(
            train_df, val_df, load_cached_data=load_cached_data
        )

        # 3. Subsample Training Data
        # We subsample AFTER global feature generation logic (which uses full data)
        # but BEFORE local feature generation (to save time)
        if len(train_df) > SUBSAMPLE_SIZE:
            print(f"Subsampling training data to {SUBSAMPLE_SIZE} rows...")
            train_df_small = train_df.sample(n=SUBSAMPLE_SIZE, random_state=SEED)
        else:
            train_df_small = train_df

        # 4. Local Features
        # generate_local_features handles caching internally
        print("Generating local features...")
        train_local = generate_local_features(
            train_df_small, "train_small", load_cached_data=load_cached_data
        )
        val_local = generate_local_features(
            val_df, "val", load_cached_data=load_cached_data
        )
        test_local = generate_local_features(
            test_df, "test", load_cached_data=load_cached_data
        )

        # 5. Merge Features
        print("Merging features...")
        # Merge Local and Global on 'key'
        # train_local contains the target 'fare_amount' and local features
        # train_global contains 'route_avg_fare'
        train_merged = train_local.merge(train_global, on="key", how="left")
        val_merged = val_local.merge(val_global, on="key", how="left")
        test_merged = test_local.merge(test_global, on="key", how="left")

        # 6. Prepare Output
        target_col = "fare_amount"
        exclude_cols = ["key", "fare_amount", "pickup_datetime"]

        feature_cols = [c for c in train_merged.columns if c not in exclude_cols]

        print(f"Final Feature Set ({len(feature_cols)}): {feature_cols}")

        X_train = train_merged[feature_cols]
        y_train = train_merged[target_col]

        X_val = val_merged[feature_cols]
        y_val = val_merged[target_col]

        X_test = test_merged[feature_cols]
        test_keys = test_merged["key"]

        return X_train, y_train, X_val, y_val, X_test, test_keys
