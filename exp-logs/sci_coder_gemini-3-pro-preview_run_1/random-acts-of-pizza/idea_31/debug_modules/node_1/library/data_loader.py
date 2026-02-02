import os
import pandas as pd
import ast
import numpy as np
from library import config


def load_dataset(split="train", load_cached_data=True):
    """
    Loads a specific dataset split ('train', 'val', or 'test') from the metadata directory.
    Parses stringified list columns (e.g., 'requester_subreddits_at_request') back to Python lists.

    Args:
        split (str): The subset to load. Options: 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from a local parquet cache.

    Returns:
        pd.DataFrame: The loaded and parsed DataFrame.
    """
    # Determine cache filename based on split and debug mode
    debug_suffix = "_debug" if config.DEBUG else ""
    cache_filename = f"{split}_parsed{debug_suffix}.parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from Metadata CSVs
    print(f"Loading {split} data from metadata CSV...")

    if split == "train":
        source_path = os.path.join(config.METADATA_DIR, "train.csv")
    elif split == "val":
        source_path = os.path.join(config.METADATA_DIR, "val.csv")
    elif split == "test":
        source_path = os.path.join(config.METADATA_DIR, "test.csv")
    else:
        raise ValueError(
            f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
        )

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    # Load CSV
    # If in debug mode, we can limit rows here for speed, though we usually filter after for consistency
    nrows = config.MAX_SAMPLES if config.DEBUG else None
    df = pd.read_csv(source_path, nrows=nrows)

    # 3. Parse Stringified Lists
    # The metadata CSVs store lists as strings (e.g., "['a', 'b']"). We need to evaluate them.
    list_columns = [
        "requester_subreddits_at_request",
        # Add other list columns if they appear in the future and are stringified
    ]

    for col in list_columns:
        if col in df.columns:
            # parsing can be slow, so we use apply with ast.literal_eval
            # Handle potential NaNs or non-string types gracefully
            df[col] = df[col].apply(
                lambda x: (
                    ast.literal_eval(x)
                    if isinstance(x, str) and x.startswith("[")
                    else (x if isinstance(x, list) else [])
                )
            )

    # 4. Save to Cache
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    try:
        df.to_parquet(cache_path, index=False)
        print(f"Cached {split} data to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache data to parquet: {e}")

    return df


def get_stratified_split(load_cached_data=True):
    """
    Retrieves the stratified training and validation sets.
    This function relies on the pre-computed splits in the metadata directory
    to ensure consistency with the project structure.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_df, val_df)
    """
    print("Retrieving stratified train/val split...")
    train_df = load_dataset("train", load_cached_data=load_cached_data)
    val_df = load_dataset("val", load_cached_data=load_cached_data)

    # Verification of split size (optional but good for sanity)
    if config.DEBUG:
        print(
            f"Debug Mode: Loaded {len(train_df)} train samples and {len(val_df)} val samples."
        )
    else:
        print(f"Loaded {len(train_df)} train samples and {len(val_df)} val samples.")

    return train_df, val_df
