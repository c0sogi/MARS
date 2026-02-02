import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    EXCLUDE_COLS,
    TARGET_COL,
    ID_COL,
)


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Implements caching to store processed NumPy arrays in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load data from cached .npy files.
                                 If False or cache miss, loads from original CSVs and updates cache.

    Returns:
        tuple: (train_data, val_data, test_data, classes)
            - train_data (dict): {'X': np.ndarray, 'y': np.ndarray, 'ids': np.ndarray}
            - val_data (dict): {'X': np.ndarray, 'y': np.ndarray, 'ids': np.ndarray}
            - test_data (dict): {'X': np.ndarray, 'ids': np.ndarray}
            - classes (np.ndarray): Sorted list of unique species names found in training data.
    """
    # Ensure working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "train_X": os.path.join(WORKING_DIR, "dataset_train_X.npy"),
        "train_y": os.path.join(WORKING_DIR, "dataset_train_y.npy"),
        "train_ids": os.path.join(WORKING_DIR, "dataset_train_ids.npy"),
        "val_X": os.path.join(WORKING_DIR, "dataset_val_X.npy"),
        "val_y": os.path.join(WORKING_DIR, "dataset_val_y.npy"),
        "val_ids": os.path.join(WORKING_DIR, "dataset_val_ids.npy"),
        "test_X": os.path.join(WORKING_DIR, "dataset_test_X.npy"),
        "test_ids": os.path.join(WORKING_DIR, "dataset_test_ids.npy"),
        "classes": os.path.join(WORKING_DIR, "dataset_classes.npy"),
    }

    # Attempt to load from cache if requested
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_paths.values())
        if all_exist:
            try:
                # Load arrays. allow_pickle=True is needed for string arrays (species names)
                train_X = np.load(cache_paths["train_X"])
                train_y = np.load(cache_paths["train_y"], allow_pickle=True)
                train_ids = np.load(cache_paths["train_ids"])

                val_X = np.load(cache_paths["val_X"])
                val_y = np.load(cache_paths["val_y"], allow_pickle=True)
                val_ids = np.load(cache_paths["val_ids"])

                test_X = np.load(cache_paths["test_X"])
                test_ids = np.load(cache_paths["test_ids"])

                classes = np.load(cache_paths["classes"], allow_pickle=True)

                return (
                    {"X": train_X, "y": train_y, "ids": train_ids},
                    {"X": val_X, "y": val_y, "ids": val_ids},
                    {"X": test_X, "ids": test_ids},
                    classes,
                )
            except Exception:
                # If loading fails (e.g. corruption), proceed to process from scratch
                pass

    # Load raw data from metadata CSVs
    # These files are guaranteed to exist by the metadata generation step
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Identify feature columns dynamically
    # Exclude ID, target, and file path columns defined in config
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS]

    # Sort feature columns to ensure deterministic order across runs
    feature_cols.sort()

    # Extract Training Data
    train_X = df_train[feature_cols].values.astype(np.float64)
    train_y = df_train[TARGET_COL].values
    train_ids = df_train[ID_COL].values

    # Extract Validation Data
    val_X = df_val[feature_cols].values.astype(np.float64)
    val_y = df_val[TARGET_COL].values
    val_ids = df_val[ID_COL].values

    # Extract Test Data
    # Test set does not have the target column
    test_X = df_test[feature_cols].values.astype(np.float64)
    test_ids = df_test[ID_COL].values

    # Extract Classes
    # Get unique species from training data and sort them alphabetically
    classes = np.unique(train_y)
    classes.sort()

    # Save processed arrays to cache
    np.save(cache_paths["train_X"], train_X)
    np.save(cache_paths["train_y"], train_y)
    np.save(cache_paths["train_ids"], train_ids)

    np.save(cache_paths["val_X"], val_X)
    np.save(cache_paths["val_y"], val_y)
    np.save(cache_paths["val_ids"], val_ids)

    np.save(cache_paths["test_X"], test_X)
    np.save(cache_paths["test_ids"], test_ids)

    np.save(cache_paths["classes"], classes)

    return (
        {"X": train_X, "y": train_y, "ids": train_ids},
        {"X": val_X, "y": val_y, "ids": val_ids},
        {"X": test_X, "ids": test_ids},
        classes,
    )
