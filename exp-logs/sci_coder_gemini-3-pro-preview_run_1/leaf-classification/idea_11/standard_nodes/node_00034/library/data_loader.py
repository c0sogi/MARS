import os
import pandas as pd
import numpy as np
from library import config


def load_dataset(split="train", load_cached_data=True, max_samples=None):
    """
    Loads the dataset for a specific split (train, val, or test), enforcing schema
    determinism and utilizing caching for efficiency.

    Args:
        split (str): The dataset split to load. Options: 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed data from
                                 the working directory cache.
        max_samples (int, optional): If provided, restricts the number of samples
                                     returned (useful for debugging).

    Returns:
        X (pd.DataFrame): The feature matrix with columns strictly ordered according
                          to config.get_ordered_feature_list().
        y (np.ndarray or None): The target labels (species). Returns None for 'test' split.
        ids (np.ndarray): The unique image identifiers.
    """
    # Ensure the working directory for this idea exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define paths for cached files
    # We use parquet for X to preserve column names and types efficiently
    # We use npy for y and ids for fast array storage
    cache_x_path = os.path.join(config.WORKING_DIR, f"X_{split}.parquet")
    cache_y_path = os.path.join(config.WORKING_DIR, f"y_{split}.npy")
    cache_ids_path = os.path.join(config.WORKING_DIR, f"ids_{split}.npy")

    # Determine if valid cache exists
    cache_exists = False
    if load_cached_data:
        # Check X and ids (required for all splits)
        if os.path.exists(cache_x_path) and os.path.exists(cache_ids_path):
            # Check y (required for train/val, not for test)
            if split in ["train", "val"]:
                if os.path.exists(cache_y_path):
                    cache_exists = True
            else:
                cache_exists = True

    if cache_exists:
        X = pd.read_parquet(cache_x_path)
        ids = np.load(cache_ids_path)
        if split in ["train", "val"]:
            y = np.load(cache_y_path, allow_pickle=True)
        else:
            y = None
    else:
        # Resolve source file path from config
        if split == "train":
            source_path = config.TRAIN_METADATA_PATH
        elif split == "val":
            source_path = config.VAL_METADATA_PATH
        elif split == "test":
            source_path = config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source metadata file not found: {source_path}")

        # Load raw data
        df = pd.read_csv(source_path)

        # Validate Schema: IDs
        if config.ID_COL not in df.columns:
            raise ValueError(
                f"Dataset {split} missing required ID column: {config.ID_COL}"
            )

        # Validate Schema: Target (for train/val)
        if split in ["train", "val"] and config.TARGET_COL not in df.columns:
            raise ValueError(
                f"Dataset {split} missing required target column: {config.TARGET_COL}"
            )

        # Validate Schema: Features
        # We strictly enforce that all expected features are present
        expected_features = config.get_ordered_feature_list()
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset {split} is missing {len(missing_features)} feature columns. "
                f"First missing: {missing_features[0]}"
            )

        # Extract and Order Data
        # Explicitly select columns to enforce deterministic order
        X = df[expected_features].copy()
        ids = df[config.ID_COL].values

        if split in ["train", "val"]:
            y = df[config.TARGET_COL].values
        else:
            y = None

        # Save to Cache
        X.to_parquet(cache_x_path)
        np.save(cache_ids_path, ids)
        if y is not None:
            np.save(cache_y_path, y)

    # Handle max_samples for debugging
    if max_samples is not None and max_samples < len(X):
        X = X.iloc[:max_samples]
        ids = ids[:max_samples]
        if y is not None:
            y = y[:max_samples]

    return X, y, ids
