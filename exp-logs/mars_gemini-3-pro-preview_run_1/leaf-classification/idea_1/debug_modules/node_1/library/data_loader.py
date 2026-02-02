import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.config import Config


def load_data(load_cached_data=True, debug=False):
    """
    Loads, preprocesses, and encodes the Leaf Classification dataset.

    This function combines the training and validation metadata files into a single
    training set for Cross-Validation. It separates features from targets, encodes
    the target labels, and prepares the test set. Results are cached to disk to
    improve efficiency in subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from the
                                 cache directory defined in Config.
        debug (bool): If True, returns a small subset of the data (defined by
                      Config.DEBUG_SAMPLES) for debugging purposes.

    Returns:
        tuple: (X, y, X_test, test_ids, label_encoder)
            - X (pd.DataFrame): The feature matrix for the full training set.
            - y (np.ndarray): The encoded target vector for the full training set.
            - X_test (pd.DataFrame): The feature matrix for the test set.
            - test_ids (np.ndarray): The IDs corresponding to the test set rows.
            - label_encoder (LabelEncoder): The fitted encoder used for species labels.
    """
    # Ensure working directory exists for caching
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths for cached artifacts
    cache_files = {
        "X": os.path.join(cache_dir, "X_train.parquet"),
        "y": os.path.join(cache_dir, "y_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.parquet"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # 1. Attempt to load from cache
    if load_cached_data:
        # Check if all required cache files exist
        if all(os.path.exists(path) for path in cache_files.values()):
            print(f"Loading processed data from cache at {cache_dir}...")
            try:
                # Load data
                X = pd.read_parquet(cache_files["X"])
                y = np.load(cache_files["y"])
                X_test = pd.read_parquet(cache_files["X_test"])
                test_ids = np.load(cache_files["test_ids"])
                classes = np.load(cache_files["classes"])

                # Reconstruct the LabelEncoder
                le = LabelEncoder()
                le.classes_ = classes

                # Apply debug sampling if requested
                if debug:
                    print(f"Debug mode: Sampling first {Config.DEBUG_SAMPLES} rows.")
                    X = X.iloc[: Config.DEBUG_SAMPLES]
                    y = y[: Config.DEBUG_SAMPLES]
                    X_test = X_test.iloc[: Config.DEBUG_SAMPLES]
                    test_ids = test_ids[: Config.DEBUG_SAMPLES]

                return X, y, X_test, test_ids, le
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing from scratch...")
        else:
            print("Cache miss. Processing data from scratch...")
    else:
        print("Force reload. Processing data from scratch...")

    # 2. Process data from scratch

    # Load metadata CSVs
    # We combine the provided 'train' and 'val' splits into one dataset
    # to allow for our own Stratified K-Fold CV strategy.
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    df_test_meta = pd.read_csv(Config.TEST_CSV)

    # Concatenate train and val
    df_full_train = pd.concat([df_train_meta, df_val_meta], axis=0, ignore_index=True)

    # Extract Targets
    y_raw = df_full_train[Config.TARGET_COL].values

    # Extract Features
    # Drop columns that are not predictive features (id, species, file paths)
    # errors='ignore' allows us to safely drop columns that might not exist in all CSVs
    X = df_full_train.drop(columns=Config.DROP_COLS, errors="ignore")

    # Prepare Test Data
    test_ids = df_test_meta[Config.ID_COL].values
    X_test = df_test_meta.drop(columns=Config.DROP_COLS, errors="ignore")

    # Encode Targets
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 3. Save to cache
    # We save the full dataset before any debug sampling ensures the cache is always complete.
    print(f"Saving processed data to cache at {cache_dir}...")
    try:
        X.to_parquet(cache_files["X"], index=False)
        np.save(cache_files["y"], y)
        X_test.to_parquet(cache_files["X_test"], index=False)
        np.save(cache_files["test_ids"], test_ids)
        np.save(cache_files["classes"], le.classes_)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    # 4. Apply debug sampling if requested
    if debug:
        print(f"Debug mode: Sampling first {Config.DEBUG_SAMPLES} rows.")
        X = X.iloc[: Config.DEBUG_SAMPLES]
        y = y[: Config.DEBUG_SAMPLES]
        X_test = X_test.iloc[: Config.DEBUG_SAMPLES]
        test_ids = test_ids[: Config.DEBUG_SAMPLES]

    return X, y, X_test, test_ids, le
