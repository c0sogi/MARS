import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


def _find_input_file(filename):
    """
    Locates a file in the input directory, checking root and subdirectories.
    """
    input_dir = "./input"
    # Check root
    path = os.path.join(input_dir, filename)
    if os.path.exists(path):
        return path

    # Check subdirectories (e.g., input/train/train.json)
    sub_dir = os.path.splitext(filename)[0]  # e.g., 'train' for 'train.json'
    path = os.path.join(input_dir, sub_dir, filename)
    if os.path.exists(path):
        return path

    return None


def load_raw_data(path):
    """
    Reads a JSON file into a Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def get_stratified_split(full_train_df):
    """
    Splits the full training data into train and validation sets
    based on the request_ids defined in the metadata CSVs.
    """
    # Load metadata to identify split IDs
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError("Metadata CSVs for splitting not found.")

    meta_train = pd.read_csv(Config.TRAIN_CSV)
    meta_val = pd.read_csv(Config.VAL_CSV)

    train_ids = set(meta_train[Config.ID_COL])
    val_ids = set(meta_val[Config.ID_COL])

    # Filter
    train_df = full_train_df[full_train_df[Config.ID_COL].isin(train_ids)].copy()
    val_df = full_train_df[full_train_df[Config.ID_COL].isin(val_ids)].copy()

    # Verify no overlap
    assert (
        len(set(train_df[Config.ID_COL]) & set(val_df[Config.ID_COL])) == 0
    ), "Overlap detected between train and val splits"

    return train_df, val_df


def load_data(load_cached_data: bool = True, debug: bool = Config.DEBUG):
    """
    Main function to load train, validation, and test datasets.
    Handles caching and debug subsampling.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed()

    # Define cache paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Loading and processing raw data...")

        # Locate raw files
        train_json_path = _find_input_file("train.json")
        test_json_path = _find_input_file("test.json")

        if not train_json_path or not test_json_path:
            raise FileNotFoundError(
                "Could not locate raw train.json or test.json in ./input"
            )

        # Load raw JSONs
        full_train_df = load_raw_data(train_json_path)
        test_df = load_raw_data(test_json_path)

        # Fix mixed types in post_was_edited for Parquet compatibility
        for df in [full_train_df, test_df]:
            if "post_was_edited" in df.columns:
                df["post_was_edited"] = df["post_was_edited"].astype(str)

        # Perform Split
        train_df, val_df = get_stratified_split(full_train_df)

        # Save to cache
        print("Saving processed data to cache...")
        # Parquet handles list columns (like subreddits) efficiently
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # Handle Debug Mode
    if debug:
        print(f"DEBUG MODE: Reducing dataset size to {Config.DEBUG_SUBSET_SIZE}")
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    print(
        f"Data Loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df
