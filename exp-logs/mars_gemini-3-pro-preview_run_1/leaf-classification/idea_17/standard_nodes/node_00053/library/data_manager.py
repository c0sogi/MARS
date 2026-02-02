import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    FEATURE_COLS,
    TARGET_COL,
    ID_COL,
    WORKING_DIR,
)
from library.utils import seed_everything


def get_data(load_cached_data=True):
    """
    Loads the training, validation, and test data.
    Implements caching using .npy files to avoid re-parsing CSVs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the working directory.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, class_names)
               X_* are float64 numpy arrays of features.
               y_* are int64 numpy arrays of class indices.
               test_ids is an int64 numpy array of image IDs.
               class_names is a numpy array of strings (the species names).
    """
    seed_everything()

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

    # Check if we can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            class_names = np.load(cache_files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, class_names
        else:
            print("Cache miss or incomplete. Processing data from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Load Metadata CSVs
    print(f"Reading {TRAIN_FILE}...")
    df_train = pd.read_csv(TRAIN_FILE)

    print(f"Reading {VAL_FILE}...")
    df_val = pd.read_csv(VAL_FILE)

    print(f"Reading {TEST_FILE}...")
    df_test = pd.read_csv(TEST_FILE)

    # Extract Features
    # We use the strictly sorted FEATURE_COLS from config to ensure deterministic ordering
    print("Extracting features...")
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    X_val = df_val[FEATURE_COLS].values.astype(np.float64)
    X_test = df_test[FEATURE_COLS].values.astype(np.float64)

    # Process Targets
    # We combine train and val species to ensure we capture all classes,
    # though stratification should guarantee this.
    # We sort unique species alphabetically to match the submission format convention.
    print("Encoding targets...")
    unique_species = sorted(df_train[TARGET_COL].unique())
    class_names = np.array(unique_species)

    # Create a mapping from species name to integer index
    species_to_idx = {species: idx for idx, species in enumerate(unique_species)}

    # Encode labels
    y_train = df_train[TARGET_COL].map(species_to_idx).values.astype(np.int64)
    y_val = df_val[TARGET_COL].map(species_to_idx).values.astype(np.int64)

    # Extract Test IDs
    test_ids = df_test[ID_COL].values.astype(np.int64)

    # Save to cache
    print("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], class_names)

    return X_train, y_train, X_val, y_val, X_test, test_ids, class_names
