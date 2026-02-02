import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import CACHE_DIR, FLOAT_PRECISION, RANDOM_SEED
from library.feature_engineering import load_and_process_data as load_raw_data

# Ensure reproducibility
np.random.seed(RANDOM_SEED)


def apply_preprocessing(X_train, X_val, X_test):
    """
    Applies the Inductive Preprocessing Pipeline:
    1. Yeo-Johnson Power Transformation (standardize=False)
    2. Standard Scaling

    Fitted ONLY on X_train. Applied to X_train, X_val, X_test.
    Maintains float64 precision.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
    """
    # 1. Power Transformation
    # standardize=False because we will apply StandardScaler explicitly afterwards
    # Yeo-Johnson supports positive and negative values.
    pt = PowerTransformer(method="yeo-johnson", standardize=False)

    # Fit on Train
    pt.fit(X_train)

    # Transform all and enforce precision
    X_train_pt = pt.transform(X_train).astype(FLOAT_PRECISION)
    X_val_pt = pt.transform(X_val).astype(FLOAT_PRECISION)
    X_test_pt = pt.transform(X_test).astype(FLOAT_PRECISION)

    # 2. Standard Scaling
    scaler = StandardScaler()

    # Fit on Train (transformed)
    scaler.fit(X_train_pt)

    # Transform all and enforce precision
    X_train_scaled = scaler.transform(X_train_pt).astype(FLOAT_PRECISION)
    X_val_scaled = scaler.transform(X_val_pt).astype(FLOAT_PRECISION)
    X_test_scaled = scaler.transform(X_test_pt).astype(FLOAT_PRECISION)

    return X_train_scaled, X_val_scaled, X_test_scaled


def load_and_merge_data(load_cached_data=True):
    """
    Wrapper around library.feature_engineering.load_and_process_data.
    Returns the raw features (Tabular + Geometric) and targets/ids.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    return load_raw_data(load_cached_data=load_cached_data)


def get_processed_data(load_cached_data=True):
    """
    Main entry point for obtaining the final preprocessed data.

    Logic:
    1. Check for cached PREPROCESSED data (npy files).
    2. If found and load_cached_data=True, load and return.
    3. If not, load RAW data (via load_and_merge_data).
    4. Apply preprocessing pipeline.
    5. Cache the preprocessed data.
    6. Return.

    Args:
        load_cached_data (bool): Whether to use cached files.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Define cache paths for transformed data
    # We use .npy for fast loading of numpy arrays
    cache_X_train_t = os.path.join(CACHE_DIR, "X_train_transformed.npy")
    cache_X_val_t = os.path.join(CACHE_DIR, "X_val_transformed.npy")
    cache_X_test_t = os.path.join(CACHE_DIR, "X_test_transformed.npy")

    files_exist = all(
        os.path.exists(f) for f in [cache_X_train_t, cache_X_val_t, cache_X_test_t]
    )

    if load_cached_data and files_exist:
        # Load transformed X
        X_train_scaled = np.load(cache_X_train_t)
        X_val_scaled = np.load(cache_X_val_t)
        X_test_scaled = np.load(cache_X_test_t)

        # Load y and ids efficiently
        # We call load_raw_data. It will load from its own cache (parquet/npy).
        # We discard the raw X returned by it, but keep y and ids.
        _, y_train, _, y_val, _, test_ids = load_raw_data(load_cached_data=True)

        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, test_ids

    # If cache miss or reload requested
    # 1. Get Raw Data (Tabular + Geometric)
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_merge_data(
        load_cached_data=load_cached_data
    )

    # 2. Apply Preprocessing Pipeline
    X_train_scaled, X_val_scaled, X_test_scaled = apply_preprocessing(
        X_train, X_val, X_test
    )

    # 3. Cache Transformed Data
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_X_train_t, X_train_scaled)
    np.save(cache_X_val_t, X_val_scaled)
    np.save(cache_X_test_t, X_test_scaled)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, test_ids
