import os
import pandas as pd
import numpy as np
from library import config


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Applies row-level cleaning to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to clean.
        is_train (bool): If True, rows outside bounds are dropped.
                         If False (Test set), coordinates are clipped to bounds to preserve row count.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Coordinate Sanitation
    # We use the bounding box defined in config
    if is_train:
        # For Training/Validation: Drop outliers
        mask = (
            (df["pickup_longitude"] >= config.MIN_LON)
            & (df["pickup_longitude"] <= config.MAX_LON)
            & (df["pickup_latitude"] >= config.MIN_LAT)
            & (df["pickup_latitude"] <= config.MAX_LAT)
            & (df["dropoff_longitude"] >= config.MIN_LON)
            & (df["dropoff_longitude"] <= config.MAX_LON)
            & (df["dropoff_latitude"] >= config.MIN_LAT)
            & (df["dropoff_latitude"] <= config.MAX_LAT)
        )
        df = df[mask]

        # 2. Fare Sanitation (Only applicable if target exists)
        if "fare_amount" in df.columns:
            fare_mask = (df["fare_amount"] >= config.MIN_FARE) & (
                df["fare_amount"] <= config.MAX_FARE
            )
            df = df[fare_mask]

    else:
        # For Test Set: Clip coordinates to bounds
        # This prevents "distance explosion" in feature engineering while keeping all keys for submission.
        df["pickup_longitude"] = df["pickup_longitude"].clip(
            config.MIN_LON, config.MAX_LON
        )
        df["pickup_latitude"] = df["pickup_latitude"].clip(
            config.MIN_LAT, config.MAX_LAT
        )
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(
            config.MIN_LON, config.MAX_LON
        )
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(
            config.MIN_LAT, config.MAX_LAT
        )

    return df


def load_and_clean(load_cached_data: bool = True):
    """
    Loads the raw data, applies cleaning/sanitation, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-cleaned data from working directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define local cache paths for the cleaned state
    cache_train_path = os.path.join(config.WORKING_DIR, "train_cleaned.parquet")
    cache_val_path = os.path.join(config.WORKING_DIR, "val_cleaned.parquet")
    cache_test_path = os.path.join(config.WORKING_DIR, "test_cleaned.parquet")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading cleaned data from cache...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)

            # Apply Debug Sampling if requested
            if config.DEBUG_SAMPLE_SIZE is not None:
                if len(train_df) > config.DEBUG_SAMPLE_SIZE:
                    print(
                        f"Sampling training data to {config.DEBUG_SAMPLE_SIZE} rows (Debug Mode)..."
                    )
                    train_df = train_df.sample(
                        n=config.DEBUG_SAMPLE_SIZE, random_state=config.SEED
                    ).reset_index(drop=True)

            return train_df, val_df, test_df

    # 2. Process from Scratch
    print("Cache not found or ignored. Loading raw metadata...")

    # Load raw files
    train_df = pd.read_parquet(config.TRAIN_META_PATH)
    val_df = pd.read_parquet(config.VAL_META_PATH)
    test_df = pd.read_parquet(config.TEST_META_PATH)

    print(f"Raw Train shape: {train_df.shape}")
    print(f"Raw Val shape: {val_df.shape}")

    # Apply Cleaning
    print("Cleaning Training data...")
    train_df = clean_dataframe(train_df, is_train=True)

    print("Cleaning Validation data...")
    val_df = clean_dataframe(val_df, is_train=True)

    print("Cleaning Test data...")
    test_df = clean_dataframe(test_df, is_train=False)

    print(f"Cleaned Train shape: {train_df.shape}")
    print(f"Cleaned Val shape: {val_df.shape}")

    # Save to Cache
    print("Saving cleaned data to cache...")
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    # Apply Debug Sampling if requested (after saving full dataset)
    if config.DEBUG_SAMPLE_SIZE is not None:
        if len(train_df) > config.DEBUG_SAMPLE_SIZE:
            print(
                f"Sampling training data to {config.DEBUG_SAMPLE_SIZE} rows (Debug Mode)..."
            )
            train_df = train_df.sample(
                n=config.DEBUG_SAMPLE_SIZE, random_state=config.SEED
            ).reset_index(drop=True)

    return train_df, val_df, test_df
