import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from library.config import (
    PATH_TRAIN,
    PATH_VAL,
    PATH_TEST,
    WORKING_DIR,
    BB_MIN_LAT,
    BB_MAX_LAT,
    BB_MIN_LON,
    BB_MAX_LON,
    FARE_MIN,
    FARE_MAX,
    META_TRAIN_SIZE,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import seed_everything, reduce_mem_usage


def load_data(path, sample_size=None):
    """
    Loads data from a parquet file. Optionally samples the data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    # Load data
    df = pd.read_parquet(path)

    # Sample if requested (and if dataset is larger than sample size)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)

    return df


def clean_data(df, is_train=True):
    """
    Applies bounding box filters and fare limits.

    Args:
        df (pd.DataFrame): Data to clean.
        is_train (bool): If True, drops rows outside bounds.
                         If False (Test), clips coordinates to bounds to preserve keys.
    """
    initial_len = len(df)

    # 1. Coordinate Sanitation
    # Define coordinate columns
    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    if is_train:
        # Strict Filtering for Train/Val: Drop rows outside bounding box
        mask = (
            (df["pickup_latitude"].between(BB_MIN_LAT, BB_MAX_LAT))
            & (df["pickup_longitude"].between(BB_MIN_LON, BB_MAX_LON))
            & (df["dropoff_latitude"].between(BB_MIN_LAT, BB_MAX_LAT))
            & (df["dropoff_longitude"].between(BB_MIN_LON, BB_MAX_LON))
        )
        df = df[mask].copy()
    else:
        # Clipping for Test: Clamp values to bounding box to preserve all rows
        df["pickup_latitude"] = df["pickup_latitude"].clip(BB_MIN_LAT, BB_MAX_LAT)
        df["pickup_longitude"] = df["pickup_longitude"].clip(BB_MIN_LON, BB_MAX_LON)
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(BB_MIN_LAT, BB_MAX_LAT)
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(BB_MIN_LON, BB_MAX_LON)

    # 2. Target Filtering (Only for data with fare_amount)
    if "fare_amount" in df.columns:
        mask_fare = df["fare_amount"].between(FARE_MIN, FARE_MAX)
        df = df[mask_fare].copy()

    # 3. Memory Optimization
    df = reduce_mem_usage(df)

    if is_train:
        print(
            f"Cleaned data: {initial_len} -> {len(df)} rows (Dropped {initial_len - len(df)})"
        )

    return df


def split_data(df):
    """
    Splits the training data into Base Train (Level 0) and Meta Train (Level 1).
    """
    train_base, train_meta = train_test_split(
        df, test_size=META_TRAIN_SIZE, random_state=SEED, shuffle=True
    )
    return train_base, train_meta


def process_data(load_cached_data=True):
    """
    Orchestrates loading, cleaning, splitting, and caching of data.

    Returns:
        tuple: (df_train_base, df_train_meta, df_val, df_test)
    """
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_train_base = os.path.join(WORKING_DIR, "train_base.parquet")
    cache_train_meta = os.path.join(WORKING_DIR, "train_meta.parquet")
    cache_val = os.path.join(WORKING_DIR, "val_processed.parquet")
    cache_test = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_base)
        and os.path.exists(cache_train_meta)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        df_train_base = pd.read_parquet(cache_train_base)
        df_train_meta = pd.read_parquet(cache_train_meta)
        df_val = pd.read_parquet(cache_val)
        df_test = pd.read_parquet(cache_test)
        return df_train_base, df_train_meta, df_val, df_test

    print("Processing data from scratch...")

    # 1. Load Raw Data
    print(f"Loading raw data (Sample Size: {DEBUG_SAMPLE_SIZE})...")
    df_train_full = load_data(PATH_TRAIN, sample_size=DEBUG_SAMPLE_SIZE)
    df_val = load_data(PATH_VAL, sample_size=DEBUG_SAMPLE_SIZE)
    df_test = load_data(
        PATH_TEST, sample_size=None
    )  # Never sample test set for final output logic

    # 2. Clean Data
    print("Cleaning Training Data...")
    df_train_full = clean_data(df_train_full, is_train=True)

    print("Cleaning Validation Data...")
    df_val = clean_data(df_val, is_train=True)

    print("Cleaning Test Data (Clipping)...")
    df_test = clean_data(df_test, is_train=False)

    # 3. Split Training Data
    print(f"Splitting Training Data (Meta Size: {META_TRAIN_SIZE})...")
    df_train_base, df_train_meta = split_data(df_train_full)

    # 4. Save to Cache
    print("Saving processed data to cache...")
    df_train_base.to_parquet(cache_train_base, index=False)
    df_train_meta.to_parquet(cache_train_meta, index=False)
    df_val.to_parquet(cache_val, index=False)
    df_test.to_parquet(cache_test, index=False)

    print("Data processing complete.")

    return df_train_base, df_train_meta, df_val, df_test
