import os
import pandas as pd
import numpy as np
from library import config
from library import utils


def load_metadata(split="train", load_cached_data=True):
    """
    Loads the metadata for a specific split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False or cache missing, loads from CSV and caches.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Determine input path based on split
    if split == "train":
        input_path = config.TRAIN_METADATA_PATH
    elif split == "val":
        input_path = config.VAL_METADATA_PATH
    elif split == "test":
        input_path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Define cache path
    cache_path = os.path.join(config.WORKING_DIR, f"metadata_{split}.parquet")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} metadata from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source CSV
    print(f"Loading {split} metadata from source: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Metadata file not found at {input_path}")

    df = pd.read_csv(input_path)

    # Optimize memory usage
    df = utils.reduce_mem_usage(df)

    # Save to cache
    print(f"Saving {split} metadata to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def load_structures(load_cached_data=True):
    """
    Loads the molecular structures data.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False or cache missing, loads from CSV and caches.

    Returns:
        pd.DataFrame: The structures dataframe containing atomic coordinates.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(config.WORKING_DIR, "structures.parquet")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading structures from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source CSV
    input_path = config.STRUCTURES_PATH
    print(f"Loading structures from source: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Structures file not found at {input_path}")

    df = pd.read_csv(input_path)

    # Optimize memory usage
    df = utils.reduce_mem_usage(df)

    # Save to cache
    print(f"Saving structures to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df
