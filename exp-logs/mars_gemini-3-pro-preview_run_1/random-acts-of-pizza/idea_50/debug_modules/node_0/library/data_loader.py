import os
import pandas as pd
from library.config import Config


def load_data(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet files to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache first.
                                 If False, forces reloading from source CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    train_cache = os.path.join(cache_dir, "train_raw.parquet")
    val_cache = os.path.join(cache_dir, "val_raw.parquet")
    test_cache = os.path.join(cache_dir, "test_raw.parquet")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Check if we should load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                print(f"Loading data from cache: {cache_dir}")
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source CSVs
    print("Loading data from metadata CSVs...")
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {Config.TRAIN_PATH}")
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(f"Validation file not found at {Config.VAL_PATH}")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"Test file not found at {Config.TEST_PATH}")

    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Ensure target column is boolean/int in train/val
    target_col = "requester_received_pizza"
    if target_col in train_df.columns:
        train_df[target_col] = train_df[target_col].astype(int)
    if target_col in val_df.columns:
        val_df[target_col] = val_df[target_col].astype(int)

    # Save to cache
    print(f"Saving data to cache: {cache_dir}")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_splits(load_cached_data: bool = True):
    """
    Wrapper around load_data that handles DEBUG sampling.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Apply Debug Sampling
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df
