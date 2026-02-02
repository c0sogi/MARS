import os
import numpy as np
import pandas as pd
from library.config import METADATA_DIR, WORKING_DIR, FEATURE_COLS, TARGET_COL, ID_COL


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism to store processed numpy arrays.
    Strictly enforces float64 precision for feature data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the working directory.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_train, X_val, X_test: np.ndarray (float64)
            - y_train, y_val: np.ndarray (string/object)
            - test_ids: np.ndarray (int/object)
            - classes: np.ndarray (sorted unique class names)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
        "classes": os.path.join(WORKING_DIR, "classes.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading cached datasets from {WORKING_DIR}...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"], allow_pickle=True)
            classes = np.load(cache_files["classes"], allow_pickle=True)

            print("Data loaded successfully from cache.")
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    # Load raw metadata
    print("Loading metadata CSVs...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Extract Features (Strict float64 enforcement)
    print(f"Extracting {len(FEATURE_COLS)} features with float64 precision...")
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    X_val = df_val[FEATURE_COLS].values.astype(np.float64)
    X_test = df_test[FEATURE_COLS].values.astype(np.float64)

    # Extract Targets
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values

    # Extract IDs for test set
    test_ids = df_test[ID_COL].values

    # Determine Classes (Sorted Alphabetically)
    # We derive classes from the training set to ensure consistency
    classes = np.unique(y_train)
    classes.sort()

    print(f"Dataset shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, ids={test_ids.shape}")
    print(f"  Classes: {len(classes)}")

    # Save to cache
    print(f"Saving processed datasets to {WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
