import os
import pandas as pd
import library.config as config
import library.feature_engineering as fe


def load_metadata(file_path):
    """
    Loads metadata from a CSV file.

    Args:
        file_path (str): Path to the metadata CSV file.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    return pd.read_csv(file_path)


def process_data_parallel(metadata_path, output_filename, load_cached_data=True):
    """
    Orchestrates the parallel feature extraction process with caching.

    This function wraps the library implementation which handles:
    1. Checking for cached Parquet files.
    2. Loading metadata.
    3. executing parallel feature extraction using ProcessPoolExecutor.
    4. Saving results to cache.

    Args:
        metadata_path (str): Path to the metadata CSV.
        output_filename (str): Name of the output parquet file (e.g., 'train_features.parquet').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # Delegate to the provided library function to avoid re-implementation
    # and ensure consistency with the provided feature_engineering logic.
    return fe.process_dataset(
        metadata_path, output_filename, load_cached_data=load_cached_data
    )


def load_data(load_cached_data=True):
    """
    Main entry point to load all datasets (Train, Val, Test).

    Args:
        load_cached_data (bool): If True, attempts to load features from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # fe.generate_train_val_test_features handles the full pipeline:
    # - identifying metadata paths from config
    # - processing/loading train, val, and test sets
    # - returning the dataframes
    return fe.generate_train_val_test_features(load_cached_data=load_cached_data)
