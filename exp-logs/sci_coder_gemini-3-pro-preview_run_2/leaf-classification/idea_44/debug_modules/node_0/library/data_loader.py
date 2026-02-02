import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    INPUT_DIR,
    FLOAT_PRECISION,
    get_cache_path,
)


def get_feature_columns(df):
    """
    Identifies and returns the list of numerical feature columns
    (margin, shape, texture) from the dataframe.
    """
    excluded = ["id", "species", "image_path"]
    return [c for c in df.columns if c not in excluded]


def _process_dataframe(df):
    """
    Internal helper to cast feature columns to the strict float precision
    defined in the configuration.
    """
    feature_cols = get_feature_columns(df)
    # Ensure features are float64 (or config precision)
    df[feature_cols] = df[feature_cols].astype(FLOAT_PRECISION)
    return df


def load_datasets(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.

    Implements caching using Parquet files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.
                                 If False, forces reloading from metadata CSVs.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    train_cache = get_cache_path("train_base.parquet")
    val_cache = get_cache_path("val_base.parquet")
    test_cache = get_cache_path("test_base.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        try:
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            return df_train, df_val, df_test
        except Exception:
            # If loading fails, fall back to raw load
            pass

    # Load from Metadata CSVs
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Process Data types
    df_train = _process_dataframe(df_train)
    df_val = _process_dataframe(df_val)
    df_test = _process_dataframe(df_test)

    # Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    return df_train, df_val, df_test


def load_image_paths(df):
    """
    Resolves the relative image paths in the dataframe to absolute paths.

    Args:
        df (pd.DataFrame): Dataframe containing the 'image_path' column.

    Returns:
        pd.Series: A series of absolute file paths.
    """
    if "image_path" not in df.columns:
        raise ValueError("Dataframe must contain 'image_path' column.")

    # Metadata contains relative paths like 'images/1.jpg'
    # We join this with the INPUT_DIR (./input)
    return df["image_path"].apply(lambda x: os.path.join(INPUT_DIR, x))
