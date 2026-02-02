import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    BB_MIN_LAT,
    BB_MAX_LAT,
    BB_MIN_LON,
    BB_MAX_LON,
    MIN_FARE,
    MAX_FARE,
    SEED,
)


def clean_training_data(df):
    """
    Filters training data by removing rows with invalid fare amounts or
    coordinates outside the specified bounding box.

    Args:
        df (pd.DataFrame): Raw training DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_len = len(df)

    # Filter fare amount
    if "fare_amount" in df.columns:
        df = df[(df["fare_amount"] >= MIN_FARE) & (df["fare_amount"] <= MAX_FARE)]

    # Filter coordinates
    # We check if coordinates are within the bounding box
    df = df[
        (df["pickup_latitude"] >= BB_MIN_LAT)
        & (df["pickup_latitude"] <= BB_MAX_LAT)
        & (df["pickup_longitude"] >= BB_MIN_LON)
        & (df["pickup_longitude"] <= BB_MAX_LON)
        & (df["dropoff_latitude"] >= BB_MIN_LAT)
        & (df["dropoff_latitude"] <= BB_MAX_LAT)
        & (df["dropoff_longitude"] >= BB_MIN_LON)
        & (df["dropoff_longitude"] <= BB_MAX_LON)
    ]

    return df


def clean_test_data(df):
    """
    Sanitizes test data by clipping coordinates to the bounding box limits.
    Does not drop rows to preserve submission format.

    Args:
        df (pd.DataFrame): Raw test DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with clipped coordinates.
    """
    # Clip coordinates to bounding box
    df["pickup_latitude"] = df["pickup_latitude"].clip(BB_MIN_LAT, BB_MAX_LAT)
    df["pickup_longitude"] = df["pickup_longitude"].clip(BB_MIN_LON, BB_MAX_LON)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(BB_MIN_LAT, BB_MAX_LAT)
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(BB_MIN_LON, BB_MAX_LON)

    return df


def get_stacking_splits(df, val_size=0.1):
    """
    Splits the training data into 'Base Train' and 'Meta Train' sets for stacking.

    Args:
        df (pd.DataFrame): The full cleaned training dataset.
        val_size (float): The proportion of the dataset to include in the Meta Train split.

    Returns:
        tuple: (base_train_df, meta_train_df)
    """
    base_train, meta_train = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True
    )
    return base_train, meta_train


def load_data(load_cached_data=True):
    """
    Loads, cleans, and returns the training, validation, and test datasets.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from processed cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # Check if cache exists and loading is requested
    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    ):
        print(f"Loading cached data from {WORKING_DIR}...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)
        return train_df, val_df, test_df

    print("Loading raw data from metadata...")
    # Load raw data
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print("Cleaning data...")
    # Apply cleaning logic
    # Train and Val get strict filtering (dropping rows)
    train_df = clean_training_data(train_df)
    val_df = clean_training_data(val_df)

    # Test gets clipping (preserving rows)
    test_df = clean_test_data(test_df)

    print(f"Saving processed data to {WORKING_DIR}...")
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Save to cache
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df
