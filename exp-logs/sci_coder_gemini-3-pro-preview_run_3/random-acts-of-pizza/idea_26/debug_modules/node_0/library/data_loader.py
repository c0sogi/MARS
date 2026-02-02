import os
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
)
from library.utils import print_header, print_info, get_cached_data


def _process_split(input_path):
    """
    Reads the metadata parquet file and removes leakage columns.

    Args:
        input_path (str): Path to the input Parquet file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print_info(f"Reading raw metadata from {input_path}...")
    df = pd.read_parquet(input_path)

    # Leakage Prevention: Drop all columns suffixed with '_at_retrieval'
    # These features are collected at the time the dataset was scraped (retrieval time),
    # not at the time of the request creation. They contain future information
    # (e.g., number of upvotes at retrieval) that constitutes leakage.
    leakage_cols = [c for c in df.columns if c.endswith("_at_retrieval")]

    if leakage_cols:
        print_info(f"Dropping {len(leakage_cols)} leakage columns: {leakage_cols}")
        df = df.drop(columns=leakage_cols)

    return df


def load_data(load_cached=True, sample_size=None):
    """
    Loads the train, validation, and test datasets.
    Applies leakage prevention by removing retrieval-time features.
    Uses caching to speed up subsequent loads.

    Args:
        load_cached (bool): If True, attempts to load processed data from cache.
        sample_size (int, optional): If provided, returns only the first N rows
                                     of each dataset for debugging purposes.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print_header("Data Loading")

    # Define cache paths for the cleaned datasets
    train_cache = os.path.join(CACHE_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_cleaned.parquet")

    # Load Train Data
    # We pass the input_path as a keyword argument which get_cached_data
    # forwards to _process_split when cache is missing.
    train_df = get_cached_data(
        cache_path=train_cache,
        process_func=_process_split,
        load_cached=load_cached,
        input_path=TRAIN_METADATA_PATH,
    )

    # Load Validation Data
    val_df = get_cached_data(
        cache_path=val_cache,
        process_func=_process_split,
        load_cached=load_cached,
        input_path=VAL_METADATA_PATH,
    )

    # Load Test Data
    test_df = get_cached_data(
        cache_path=test_cache,
        process_func=_process_split,
        load_cached=load_cached,
        input_path=TEST_METADATA_PATH,
    )

    # Handle subsampling for debugging
    if sample_size is not None:
        print_info(f"Subsampling datasets to {sample_size} rows for debugging...")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    print_info(f"Train shape: {train_df.shape}")
    print_info(f"Val shape:   {val_df.shape}")
    print_info(f"Test shape:  {test_df.shape}")

    return train_df, val_df, test_df
