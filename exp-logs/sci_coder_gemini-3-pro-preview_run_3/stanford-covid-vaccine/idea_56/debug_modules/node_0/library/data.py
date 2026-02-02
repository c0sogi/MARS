import os
import pandas as pd
from library.config import Config, RNADataset, get_data_loader, load_and_cache_data


def load_data(debug: bool = False, load_cached_data: bool = True):
    """
    Loads the training, validation, and test dataframes.

    Implements the caching logic by delegating to library.config.load_and_cache_data,
    which checks for existing parquet files in the working directory before loading
    from the metadata source.

    Args:
        debug (bool): If True, subsamples the datasets for rapid debugging.
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache missing, reloads from source and caches.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure the working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load data using the provided utility from config
    # This handles the check for cached files and saving if necessary
    train_df, val_df, test_df = load_and_cache_data(load_cached_data=load_cached_data)

    if debug:
        # Subsample data for debugging purposes
        # Using a small multiple of batch size to ensure valid batches
        subset_size = Config.BATCH_SIZE * 2
        train_df = train_df.iloc[:subset_size].reset_index(drop=True)
        val_df = val_df.iloc[:subset_size].reset_index(drop=True)
        test_df = test_df.iloc[:subset_size].reset_index(drop=True)
        print(f"Debug mode enabled: Data subsampled to {subset_size} rows.")

    return train_df, val_df, test_df


def get_loaders(debug: bool = False, load_cached_data: bool = True):
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses subsampled data.
        load_cached_data (bool): If True, uses cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load the dataframes
    train_df, val_df, test_df = load_data(
        debug=debug, load_cached_data=load_cached_data
    )

    # Create DataLoaders using the utility from config
    # Train loader is shuffled, Val/Test are not
    train_loader = get_data_loader(
        train_df, mode="train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_data_loader(
        val_df, mode="val", batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = get_data_loader(
        test_df, mode="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader
