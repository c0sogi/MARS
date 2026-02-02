import os
import pandas as pd
from library import config
from library import data_loader


def clamp_coordinates(df):
    """
    Restricts pickup and dropoff coordinates to the NYC bounding box defined in config.
    Values outside the bounding box are clipped to the boundaries.
    """
    min_lon, max_lon, min_lat, max_lat = config.NYC_BOUNDING_BOX

    # Clip Longitudes
    if "pickup_longitude" in df.columns:
        df["pickup_longitude"] = df["pickup_longitude"].clip(min_lon, max_lon)
    if "dropoff_longitude" in df.columns:
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(min_lon, max_lon)

    # Clip Latitudes
    if "pickup_latitude" in df.columns:
        df["pickup_latitude"] = df["pickup_latitude"].clip(min_lat, max_lat)
    if "dropoff_latitude" in df.columns:
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(min_lat, max_lat)

    return df


def round_coordinates(df):
    """
    Rounds coordinate columns to the precision specified in config.
    This performs spatial discretization.
    """
    precision = config.COORD_PRECISION
    cols_to_round = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    # Only round columns that exist in the dataframe
    target_cols = [c for c in cols_to_round if c in df.columns]

    if target_cols:
        df[target_cols] = df[target_cols].round(precision)

    return df


def filter_target_outliers(df):
    """
    Removes rows with fare_amount outliers.
    Cite solution_lesson_node_00017: Target Variable Sanitization.
    """
    if config.TARGET_COL in df.columns:
        initial_len = len(df)
        df = df[
            (df[config.TARGET_COL] >= config.MIN_FARE)
            & (df[config.TARGET_COL] <= config.MAX_FARE)
        ].copy()
        print(f"Filtered target outliers: {initial_len} -> {len(df)} rows")
    return df


def preprocess_dataframe(df):
    """
    Applies the full preprocessing pipeline (clamping then rounding) to a dataframe.
    """
    df = clamp_coordinates(df)
    df = round_coordinates(df)
    return df


def get_preprocessed_splits(load_cached_data=True):
    """
    Loads data splits, applies preprocessing, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (full_train_df, train_subsample_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "full": os.path.join(config.WORKING_DIR, "processed_full_train.parquet"),
        "sub": os.path.join(config.WORKING_DIR, "processed_train_subsample.parquet"),
        "val": os.path.join(config.WORKING_DIR, "processed_val.parquet"),
        "test": os.path.join(config.WORKING_DIR, "processed_test.parquet"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_paths.values())

    if load_cached_data and all_cached:
        # Load from cache
        full_train_df = pd.read_parquet(cache_paths["full"], engine="pyarrow")
        train_subsample_df = pd.read_parquet(cache_paths["sub"], engine="pyarrow")
        val_df = pd.read_parquet(cache_paths["val"], engine="pyarrow")
        test_df = pd.read_parquet(cache_paths["test"], engine="pyarrow")

        return full_train_df, train_subsample_df, val_df, test_df

    else:
        # Load raw data using data_loader
        # Pass load_cached_data to data_loader to utilize its internal caching for the raw subsample
        raw_full, raw_sub, raw_val, raw_test = data_loader.get_data_splits(
            load_cached_data=load_cached_data
        )

        # Process datasets
        full_train_df = preprocess_dataframe(raw_full)
        train_subsample_df = preprocess_dataframe(raw_sub)
        val_df = preprocess_dataframe(raw_val)
        test_df = preprocess_dataframe(raw_test)

        # Save to cache
        full_train_df.to_parquet(cache_paths["full"], index=False)
        train_subsample_df.to_parquet(cache_paths["sub"], index=False)
        val_df.to_parquet(cache_paths["val"], index=False)
        test_df.to_parquet(cache_paths["test"], index=False)

        return full_train_df, train_subsample_df, val_df, test_df
