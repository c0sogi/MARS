import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    IDEA_DIR,
    FEATURE_COLS,
    TARGET_COL,
    ID_COL,
)


def load_datasets(load_cached_data: bool = True):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism using Parquet for feature DataFrames and
    Numpy .npy files for targets/IDs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the working directory.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            - X_train, X_val, X_test: pd.DataFrame containing float64 features.
            - y_train, y_val: np.ndarray containing target labels.
            - test_ids: np.ndarray containing test image IDs.
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(IDEA_DIR, "X_train.parquet"),
        "y_train": os.path.join(IDEA_DIR, "y_train.npy"),
        "X_val": os.path.join(IDEA_DIR, "X_val.parquet"),
        "y_val": os.path.join(IDEA_DIR, "y_val.npy"),
        "X_test": os.path.join(IDEA_DIR, "X_test.parquet"),
        "test_ids": os.path.join(IDEA_DIR, "test_ids.npy"),
    }

    # Ensure working directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        try:
            print(f"Attempting to load cached data from {IDEA_DIR}...")
            if all(os.path.exists(path) for path in cache_files.values()):
                X_train = pd.read_parquet(cache_files["X_train"])
                y_train = np.load(cache_files["y_train"], allow_pickle=True)
                X_val = pd.read_parquet(cache_files["X_val"])
                y_val = np.load(cache_files["y_val"], allow_pickle=True)
                X_test = pd.read_parquet(cache_files["X_test"])
                test_ids = np.load(cache_files["test_ids"], allow_pickle=True)

                print("Successfully loaded data from cache.")
                return X_train, y_train, X_val, y_val, X_test, test_ids
            else:
                print("Cache incomplete or missing. Reloading from source...")
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from source...")

    # Load from source metadata CSVs
    print("Loading raw data from metadata CSVs...")

    # Read CSVs
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Process Training Data
    # Select specific features to ensure deterministic ordering
    X_train = df_train[FEATURE_COLS].astype("float64")
    y_train = df_train[TARGET_COL].values

    # Process Validation Data
    X_val = df_val[FEATURE_COLS].astype("float64")
    y_val = df_val[TARGET_COL].values

    # Process Test Data
    X_test = df_test[FEATURE_COLS].astype("float64")
    test_ids = df_test[ID_COL].values

    # Save to cache
    print(f"Saving processed data to cache at {IDEA_DIR}...")
    try:
        X_train.to_parquet(cache_files["X_train"])
        np.save(cache_files["y_train"], y_train)

        X_val.to_parquet(cache_files["X_val"])
        np.save(cache_files["y_val"], y_val)

        X_test.to_parquet(cache_files["X_test"])
        np.save(cache_files["test_ids"], test_ids)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
