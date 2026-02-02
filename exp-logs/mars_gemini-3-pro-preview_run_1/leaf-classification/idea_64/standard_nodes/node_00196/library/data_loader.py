import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    ID_COL,
    TARGET_COL,
    GEOMETRIC_FEATURES,
    TABULAR_PREFIXES,
)
from library.image_processing import process_images


def load_and_process_data(load_cached_data=True):
    """
    Loads metadata, extracts/loads geometric features, merges with tabular features,
    enforces alphanumeric column sorting, and returns processed numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load merged data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # 1. Try Loading from Cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading merged feature matrices from cache...")
        try:
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            return X_train, y_train, X_val, y_val, X_test, test_ids
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing from scratch...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_DATA_PATH}")

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # 3. Extract Geometric Features
    # process_images handles caching of the geometric features specifically
    print("Retrieving geometric features...")
    geo_train = process_images(df_train, "train", load_cached_data)
    geo_val = process_images(df_val, "val", load_cached_data)
    geo_test = process_images(df_test, "test", load_cached_data)

    # 4. Feature Fusion and Deterministic Ordering
    def merge_features(df_meta, geo_array):
        """
        Merges tabular features from metadata with geometric array.
        Enforces alphanumeric column ordering.
        """
        # Identify tabular columns (margin_*, shape_*, texture_*)
        tabular_cols = [
            c
            for c in df_meta.columns
            if any(c.startswith(prefix) for prefix in TABULAR_PREFIXES)
        ]

        # Create DataFrames
        df_tabular = df_meta[tabular_cols].copy()
        df_geo = pd.DataFrame(geo_array, columns=GEOMETRIC_FEATURES)

        # Reset indices to ensure safe concatenation
        df_tabular.reset_index(drop=True, inplace=True)
        df_geo.reset_index(drop=True, inplace=True)

        # Concatenate
        df_combined = pd.concat([df_tabular, df_geo], axis=1)

        # Enforce Alphanumeric Column Ordering
        # This ensures 'Area' comes before 'margin_1', etc., in a deterministic way
        sorted_cols = sorted(df_combined.columns)
        df_combined = df_combined.reindex(columns=sorted_cols)

        # Return as float64 numpy array
        return df_combined.values.astype(np.float64)

    print("Merging features and enforcing column order...")
    X_train = merge_features(df_train, geo_train)
    X_val = merge_features(df_val, geo_val)
    X_test = merge_features(df_test, geo_test)

    # Extract Targets and IDs
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    test_ids = df_test[ID_COL].values

    # 5. Save to Cache
    print("Saving merged datasets to cache...")
    try:
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["test_ids"], test_ids)
    except Exception as e:
        print(f"Warning: Could not save merged cache: {e}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
