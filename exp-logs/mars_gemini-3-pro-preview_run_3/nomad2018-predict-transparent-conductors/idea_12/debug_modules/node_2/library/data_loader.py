import os
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.descriptors import extract_features


def load_data(
    load_cached_data: bool = True,
    debug: bool = DEBUG,
    sample_size: int = DEBUG_SAMPLE_SIZE,
):
    """
    Loads data, generates or loads structural features (RDF/ADF), and returns processed DataFrames.

    Args:
        load_cached_data (bool): If True, attempts to load features from parquet cache.
        debug (bool): If True, processes only a small subset of the data.
        sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"Loading metadata from {os.path.dirname(TRAIN_METADATA_PATH)}...")
    train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(VAL_METADATA_PATH)
    test_meta = pd.read_csv(TEST_METADATA_PATH)

    # Determine cache paths based on debug mode to avoid cache pollution
    if debug:
        print(f"Debug mode enabled. Sampling {sample_size} rows per dataset.")
        train_meta = train_meta.iloc[:sample_size]
        val_meta = val_meta.iloc[:sample_size]
        test_meta = test_meta.iloc[:sample_size]

        train_cache = TRAIN_FEATURES_PATH.replace(".parquet", "_debug.parquet")
        val_cache = VAL_FEATURES_PATH.replace(".parquet", "_debug.parquet")
        test_cache = TEST_FEATURES_PATH.replace(".parquet", "_debug.parquet")
    else:
        train_cache = TRAIN_FEATURES_PATH
        val_cache = VAL_FEATURES_PATH
        test_cache = TEST_FEATURES_PATH

    print("Processing Training Set...")
    train_df = extract_features(
        metadata_df=train_meta,
        load_cached_data=load_cached_data,
        cache_path=train_cache,
    )

    print("Processing Validation Set...")
    val_df = extract_features(
        metadata_df=val_meta, load_cached_data=load_cached_data, cache_path=val_cache
    )

    print("Processing Test Set...")
    test_df = extract_features(
        metadata_df=test_meta, load_cached_data=load_cached_data, cache_path=test_cache
    )

    print(f"Data loading complete.")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    return train_df, val_df, test_df
