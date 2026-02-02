import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, feature_engineering


def load_dataset(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Delegates to library.feature_engineering.get_processed_data to:
    1. Load metadata.
    2. Extract geometric features from images.
    3. Merge with tabular features.
    4. Cache the raw feature matrices.

    Args:
        load_cached_data (bool): Whether to attempt loading from the feature engineering cache.

    Returns:
        tuple:
            (X_train, y_train, ids_train),
            (X_val, y_val, ids_val),
            (X_test, ids_test)
    """
    # Load Training Data
    X_train, y_train, ids_train = feature_engineering.get_processed_data(
        config.TRAIN_DATA_PATH, "train", load_cached_data=load_cached_data
    )

    # Load Validation Data
    X_val, y_val, ids_val = feature_engineering.get_processed_data(
        config.VAL_DATA_PATH, "val", load_cached_data=load_cached_data
    )

    # Load Test Data
    X_test, _, ids_test = feature_engineering.get_processed_data(
        config.TEST_DATA_PATH, "test", load_cached_data=load_cached_data
    )

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)


def preprocess_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the High-Precision Pipeline to the feature matrices.

    Pipeline Steps:
    1. Enforce alphanumeric column ordering (if DataFrame).
    2. Cast to float64 (Double Precision).
    3. Apply Yeo-Johnson Power Transformation (standardize=False).
    4. Apply Standard Scaling.

    Constraints:
    - Transformers are fitted ONLY on X_train to prevent data leakage.
    - Results are cached to disk as .npy files.

    Args:
        X_train (pd.DataFrame or np.ndarray): Raw training features.
        X_val (pd.DataFrame or np.ndarray): Raw validation features.
        X_test (pd.DataFrame or np.ndarray): Raw test features.
        load_cached_data (bool): Whether to load preprocessed arrays from cache if available.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled) as float64 numpy arrays.
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_train = os.path.join(cache_dir, "X_train_preprocessed.npy")
    path_val = os.path.join(cache_dir, "X_val_preprocessed.npy")
    path_test = os.path.join(cache_dir, "X_test_preprocessed.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(path_train)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
        ):
            X_train_scaled = np.load(path_train)
            X_val_scaled = np.load(path_val)
            X_test_scaled = np.load(path_test)
            return X_train_scaled, X_val_scaled, X_test_scaled

    # 2. Process from Scratch

    # Ensure Determinism: Sort columns if input is DataFrame
    if isinstance(X_train, pd.DataFrame):
        cols = sorted(X_train.columns)
        X_train = X_train[cols].values
        X_val = X_val[cols].values
        X_test = X_test[cols].values

    # Enforce High Precision (float64)
    X_train = X_train.astype(config.FLOAT_TYPE)
    X_val = X_val.astype(config.FLOAT_TYPE)
    X_test = X_test.astype(config.FLOAT_TYPE)

    # Initialize Transformers
    # Yeo-Johnson stabilizes variance; standardize=False because we use StandardScaler next
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    scaler = StandardScaler()

    # Inductive Fitting: Fit ONLY on Training Data
    pt.fit(X_train)

    # Transform all sets with the learned PowerTransformer
    X_train_pt = pt.transform(X_train)
    X_val_pt = pt.transform(X_val)
    X_test_pt = pt.transform(X_test)

    # Fit StandardScaler ONLY on the transformed Training Data
    scaler.fit(X_train_pt)

    # Transform all sets with the learned Scaler
    X_train_scaled = scaler.transform(X_train_pt)
    X_val_scaled = scaler.transform(X_val_pt)
    X_test_scaled = scaler.transform(X_test_pt)

    # 3. Save to Cache
    np.save(path_train, X_train_scaled)
    np.save(path_val, X_val_scaled)
    np.save(path_test, X_test_scaled)

    return X_train_scaled, X_val_scaled, X_test_scaled
