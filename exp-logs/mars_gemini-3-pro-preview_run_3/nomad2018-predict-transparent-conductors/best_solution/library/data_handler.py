import os
import random
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEBUG_SAMPLE_SIZE,
    RANDOM_SEED,
)
from library.feature_engine import process_dataset

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _load_and_process(metadata_path, load_cached_data=True, sample_size=None):
    """
    Helper function to load metadata CSV, optionally sample it, and then
    process it using the feature engine to generate/load features.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): Number of samples to load. If None,
                                     defaults to DEBUG_SAMPLE_SIZE from config.

    Returns:
        pd.DataFrame: Processed dataframe with features merged.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Determine effective sample size
    # If sample_size is provided, use it. Otherwise fall back to config.
    n_samples = sample_size if sample_size is not None else DEBUG_SAMPLE_SIZE

    # Apply sampling if required
    if n_samples is not None and n_samples < len(df):
        print(
            f"Sampling {n_samples} examples from {len(df)} available in {os.path.basename(metadata_path)}..."
        )
        df = df.sample(n=n_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    # Delegate processing to feature_engine
    # process_dataset handles iteration over rows, loading .xyz files,
    # generating fingerprints, caching, and merging tabular data.
    df_processed = process_dataset(df, load_cached_data=load_cached_data)

    return df_processed


def get_train_data(load_cached_data=True, sample_size=None):
    """
    Loads and processes the training dataset.
    """
    print(f"--- Loading Training Data ---")
    return _load_and_process(TRAIN_METADATA_PATH, load_cached_data, sample_size)


def get_val_data(load_cached_data=True, sample_size=None):
    """
    Loads and processes the validation dataset.
    """
    print(f"--- Loading Validation Data ---")
    return _load_and_process(VAL_METADATA_PATH, load_cached_data, sample_size)


def get_test_data(load_cached_data=True, sample_size=None):
    """
    Loads and processes the test dataset.
    """
    print(f"--- Loading Test Data ---")
    return _load_and_process(TEST_METADATA_PATH, load_cached_data, sample_size)
