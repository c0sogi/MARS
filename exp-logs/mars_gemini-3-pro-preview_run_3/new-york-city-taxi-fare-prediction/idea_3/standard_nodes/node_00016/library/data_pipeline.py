import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SEED,
    MIN_FARE,
    MAX_FARE,
)
from library.utils import rotate_coordinates, add_landmark_features
from library.spatial_encoder import process_spatial_features


def extract_temporal_features(df):
    """
    Converts pickup_datetime to datetime objects and extracts
    hour, year, and day_of_week.
    """
    # Ensure datetime format
    if df["pickup_datetime"].dtype == "object":
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    return df


def apply_geometric_features(df):
    """
    Applies coordinate rotation and landmark distance calculations.
    """
    df = rotate_coordinates(df)
    df = add_landmark_features(df)
    return df


def load_and_process(load_cached_data=True, debug_sample_size=None):
    """
    Main data processing pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        debug_sample_size (int or None): If set, samples the training data for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Define cache paths for the final fully processed datasets
    # Note: process_spatial_features handles its own caching for the intermediate spatial step.
    # We add a suffix if debugging to avoid overwriting full data cache with sample cache.
    suffix = "" if debug_sample_size is None else f"_sample_{debug_sample_size}"

    train_cache_path = os.path.join(CACHE_DIR, f"train_final{suffix}.parquet")
    val_cache_path = os.path.join(CACHE_DIR, f"val_final{suffix}.parquet")
    test_cache_path = os.path.join(CACHE_DIR, f"test_final{suffix}.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print(f"Loading final processed data from cache (suffix='{suffix}')...")
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            test_df = pd.read_parquet(test_cache_path)

            # Prepare outputs
            y_train = train_df["fare_amount"]
            X_train = train_df.drop(
                columns=["fare_amount", "key", "pickup_datetime"], errors="ignore"
            )

            y_val = val_df["fare_amount"]
            X_val = val_df.drop(
                columns=["fare_amount", "key", "pickup_datetime"], errors="ignore"
            )

            test_ids = test_df["key"]
            X_test = test_df.drop(columns=["key", "pickup_datetime"], errors="ignore")

            return X_train, y_train, X_val, y_val, X_test, test_ids

    # 2. Load Raw Data
    print("Loading raw data from metadata...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Filter Validation Data (Cite solution_lesson_node_00005)
    # Removing outliers from validation set is crucial for accurate RMSE evaluation
    print(f"Filtering validation data (Range: {MIN_FARE}-{MAX_FARE})...")
    val_mask = (val_df["fare_amount"] >= MIN_FARE) & (val_df["fare_amount"] <= MAX_FARE)
    val_df = val_df[val_mask].reset_index(drop=True)

    # 3. Sampling (if debugging)
    if debug_sample_size is not None and len(train_df) > debug_sample_size:
        print(f"Sampling training data to {debug_sample_size} rows...")
        # Simple random sampling
        train_df = train_df.sample(n=debug_sample_size, random_state=SEED).reset_index(
            drop=True
        )

    # 4. Spatial Processing (Integration)
    # This function handles:
    #   - Filtering train_df for outliers (MIN_FARE <= fare <= MAX_FARE)
    #   - Computing spatial clusters and target encoding
    #   - Returning dataframes with new spatial columns
    # We pass load_cached_data=False if we are sampling, to force re-computation on the sample
    spatial_cache_flag = load_cached_data and (debug_sample_size is None)
    train_df, val_df, test_df = process_spatial_features(
        train_df, val_df, test_df, load_cached_data=spatial_cache_flag
    )

    # 5. Feature Engineering (Temporal & Geometric)
    print("Applying temporal and geometric feature engineering...")

    for df in [train_df, val_df, test_df]:
        extract_temporal_features(df)
        apply_geometric_features(df)

    # 6. Save to Cache
    print("Saving fully processed data to cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    # 7. Prepare Return Values
    # Drop non-feature columns for X matrices
    drop_cols_train = ["fare_amount", "key", "pickup_datetime"]
    drop_cols_test = ["key", "pickup_datetime"]

    y_train = train_df["fare_amount"]
    X_train = train_df.drop(columns=drop_cols_train, errors="ignore")

    y_val = val_df["fare_amount"]
    X_val = val_df.drop(columns=drop_cols_train, errors="ignore")

    test_ids = test_df["key"]
    X_test = test_df.drop(columns=drop_cols_test, errors="ignore")

    return X_train, y_train, X_val, y_val, X_test, test_ids
