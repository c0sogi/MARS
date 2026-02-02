import os
import pandas as pd
import numpy as np
from library.config import (
    WORKING_DIR,
    FLOAT_PRECISION,
    SORT_COLUMNS,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)
from library.feature_extraction import process_dataset


def load_dataset(dataset_name, csv_path, load_cached_data=True, limit=None):
    """
    Loads a specific dataset split, merging tabular metadata with extracted geometric features.
    Handles caching of the merged result to parquet/npy files.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        csv_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.
        limit (int, optional): Limit number of rows for debugging.

    Returns:
        tuple: (X, y, ids) where X is a DataFrame, y is a numpy array (or None), ids is a numpy array.
    """
    # Define cache paths
    cache_X_path = os.path.join(WORKING_DIR, f"X_{dataset_name}.parquet")
    cache_y_path = os.path.join(WORKING_DIR, f"y_{dataset_name}.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"ids_{dataset_name}.npy")

    # Attempt to load from cache
    if load_cached_data and limit is None:
        # We check for X and ids. y is optional (e.g. test set might not have it, or it might be cached)
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            try:
                print(f"Loading merged {dataset_name} data from cache...")
                X = pd.read_parquet(cache_X_path)
                ids = np.load(cache_ids_path, allow_pickle=True)

                if os.path.exists(cache_y_path):
                    y = np.load(cache_y_path, allow_pickle=True)
                else:
                    y = None
                return X, y, ids
            except Exception as e:
                print(f"Error loading cache for {dataset_name}: {e}. Recomputing...")
        else:
            # Cache missing, proceed to compute
            pass

    print(f"Processing and merging data for {dataset_name}...")

    # 1. Load Metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df_meta = pd.read_csv(csv_path)
    if limit is not None:
        df_meta = df_meta.head(limit)

    # 2. Get Geometric Features
    # process_dataset handles extraction and its own caching of the raw geometric features
    df_geo = process_dataset(
        csv_path, dataset_name, load_cached_data=load_cached_data, limit=limit
    )

    # 3. Merge DataFrames
    # df_meta contains 'id', 'species', 'file_path', and original features
    # df_geo contains geometric features, indexed by 'id'

    if "id" not in df_meta.columns:
        raise ValueError(f"Column 'id' missing in {csv_path}")

    # Perform left merge on 'id'
    df_merged = df_meta.merge(df_geo, left_on="id", right_index=True, how="left")

    # 4. Separate Components
    ids = df_merged["id"].values

    if "species" in df_merged.columns:
        y = df_merged["species"].values
    else:
        y = None

    # Filter feature columns
    # Exclude non-feature columns
    non_feature_cols = {"id", "species", "file_path", "full_path"}
    feature_cols = [c for c in df_merged.columns if c not in non_feature_cols]

    X = df_merged[feature_cols]

    # 5. Post-Processing
    # Enforce float precision
    X = X.astype(FLOAT_PRECISION)

    # Enforce alphanumeric column ordering
    if SORT_COLUMNS:
        X = X.sort_index(axis=1)

    # 6. Save to Cache
    # Only cache if we are not debugging with a limit
    if limit is None:
        print(f"Saving merged {dataset_name} data to cache...")
        X.to_parquet(cache_X_path)
        np.save(cache_ids_path, ids)
        if y is not None:
            np.save(cache_y_path, y)

    return X, y, ids


def get_data_loaders(load_cached_data=True, limit=None):
    """
    Main entry point to retrieve all dataset splits (Train, Val, Test).

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        limit (int, optional): Limit samples for debugging.

    Returns:
        tuple: (train_data, val_data, test_data)
        Each element is a tuple (X, y, ids).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_data = load_dataset("train", TRAIN_CSV, load_cached_data, limit)
    val_data = load_dataset("val", VAL_CSV, load_cached_data, limit)
    test_data = load_dataset("test", TEST_CSV, load_cached_data, limit)

    return train_data, val_data, test_data
