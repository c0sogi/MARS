import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import (
    CACHE_DIR,
    TARGET_COL,
    ID_COL,
    PRECISION_TYPE,
    get_all_feature_names,
)
from library.feature_engineering import process_data
from library.utils import validate_precision


def load_and_augment_data(load_cached_data=True):
    """
    Loads the dataset, augments it with geometric features, enforces alphanumeric
    feature ordering and double precision, and prepares the final numpy arrays
    for modeling.

    Implements a caching mechanism using .npy files to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the cache directory.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features (float64).
            - y_train (np.ndarray): Encoded training targets (int).
            - X_val (np.ndarray): Validation features (float64).
            - y_val (np.ndarray): Encoded validation targets (int).
            - X_test (np.ndarray): Test features (float64).
            - test_ids (np.ndarray): Test image IDs.
            - classes (np.ndarray): Array of original class names (strings).
    """
    # Define cache file paths
    path_X_train = os.path.join(CACHE_DIR, "X_train.npy")
    path_y_train = os.path.join(CACHE_DIR, "y_train.npy")
    path_X_val = os.path.join(CACHE_DIR, "X_val.npy")
    path_y_val = os.path.join(CACHE_DIR, "y_val.npy")
    path_X_test = os.path.join(CACHE_DIR, "X_test.npy")
    path_ids_test = os.path.join(CACHE_DIR, "test_ids.npy")
    path_classes = os.path.join(CACHE_DIR, "classes.npy")

    cache_files = [
        path_X_train,
        path_y_train,
        path_X_val,
        path_y_val,
        path_X_test,
        path_ids_test,
        path_classes,
    ]

    # --- 1. Attempt to Load from Cache ---
    if load_cached_data and all(os.path.exists(p) for p in cache_files):
        print("Loading processed numpy arrays from cache...")
        try:
            X_train = np.load(path_X_train)
            y_train = np.load(path_y_train)
            X_val = np.load(path_X_val)
            y_val = np.load(path_y_val)
            X_test = np.load(path_X_test)
            test_ids = np.load(path_ids_test)
            classes = np.load(path_classes, allow_pickle=True)

            # Validate precision to ensure cache integrity
            validate_precision(X_train, "X_train (Cache)")
            validate_precision(X_val, "X_val (Cache)")
            validate_precision(X_test, "X_test (Cache)")

            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # --- 2. Process Data from Scratch ---
    print("Processing data from feature engineering pipeline...")

    # Load and augment DataFrames (this step handles parquet caching internally)
    df_train, df_val, df_test = process_data(load_cached_data=load_cached_data)

    # Get the deterministic list of feature names (Tabular + Geometric)
    # Sorted alphanumerically to ensure consistent column ordering
    feature_cols = get_all_feature_names()
    print(f"Constructing feature matrices with {len(feature_cols)} features...")

    # Verify features exist in the dataframe
    missing_cols = [c for c in feature_cols if c not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing features in training data: {missing_cols}")

    # --- 3. Extract and Cast Features (X) ---
    # We strictly enforce PRECISION_TYPE (float64) here
    X_train = df_train[feature_cols].values.astype(PRECISION_TYPE)
    X_val = df_val[feature_cols].values.astype(PRECISION_TYPE)
    X_test = df_test[feature_cols].values.astype(PRECISION_TYPE)

    validate_precision(X_train, "X_train")
    validate_precision(X_val, "X_val")
    validate_precision(X_test, "X_test")

    # --- 4. Encode Targets (y) ---
    print("Encoding target labels...")
    le = LabelEncoder()
    # Fit on training data
    y_train = le.fit_transform(df_train[TARGET_COL])
    # Transform validation data (stratified split ensures coverage, but handle carefully)
    y_val = le.transform(df_val[TARGET_COL])
    classes = le.classes_

    # --- 5. Extract IDs ---
    test_ids = df_test[ID_COL].values

    # --- 6. Save to Cache ---
    print("Saving processed numpy arrays to cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(path_X_train, X_train)
    np.save(path_y_train, y_train)
    np.save(path_X_val, X_val)
    np.save(path_y_val, y_val)
    np.save(path_X_test, X_test)
    np.save(path_ids_test, test_ids)
    np.save(path_classes, classes)

    print("Data loading and augmentation complete.")
    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
