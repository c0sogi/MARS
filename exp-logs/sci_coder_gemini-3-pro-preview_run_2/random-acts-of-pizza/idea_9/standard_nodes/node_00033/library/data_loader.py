import os
import pandas as pd
from library.utils import load_raw_data


def load_dataset(
    metadata_dir="./metadata",
    input_dir="./input",
    load_cached_data=True,
    debug_sample_size=None,
):
    """
    Loads the dataset, merging metadata with raw JSON data.
    Implements caching to parquet files and debug subsampling.

    Args:
        metadata_dir (str): Path to metadata directory.
        input_dir (str): Path to input directory.
        load_cached_data (bool): Whether to try loading from cache.
        debug_sample_size (int, optional): Number of samples to return for debugging.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    cache_dir = "./working/idea_9"
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "df_train.parquet")
    val_cache = os.path.join(cache_dir, "df_val.parquet")
    test_cache = os.path.join(cache_dir, "df_test.parquet")

    files_exist = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    # Try loading from cache
    data_loaded = False
    if load_cached_data and files_exist:
        try:
            print("Loading datasets from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            data_loaded = False

    # Compute from scratch if needed
    if not data_loaded:
        print("Loading datasets from raw files...")
        df_train, df_val, df_test = load_raw_data(
            metadata_dir=metadata_dir, input_dir=input_dir
        )

        # Save to cache
        print("Saving datasets to cache...")
        try:
            df_train.to_parquet(train_cache, index=False)
            df_val.to_parquet(val_cache, index=False)
            df_test.to_parquet(test_cache, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # Apply debug subsampling if requested
    if debug_sample_size is not None:
        print(f"Subsampling datasets to {debug_sample_size} samples...")
        df_train = df_train.iloc[:debug_sample_size].copy()
        df_val = df_val.iloc[:debug_sample_size].copy()
        df_test = df_test.iloc[:debug_sample_size].copy()

    return df_train, df_val, df_test
