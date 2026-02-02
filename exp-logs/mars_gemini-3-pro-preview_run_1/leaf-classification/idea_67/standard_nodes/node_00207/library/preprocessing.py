import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler

from library.config import (
    CACHE_DIR,
    FLOAT_PRECISION,
    VARIANCE_THRESHOLD,
    APPLY_YEO_JOHNSON,
    APPLY_SCALING,
)
from library.utils import set_seed


class RobustPreprocessor:
    """
    A robust preprocessing pipeline that applies:
    1. VarianceThreshold (Sanitization)
    2. Yeo-Johnson Power Transformation (Gaussianization)
    3. StandardScaler (Normalization)

    Operates strictly in float64 precision.
    """

    def __init__(self):
        self.vt = None
        self.pt = None
        self.scaler = None

    def fit(self, X):
        """
        Fits the preprocessing pipeline on the training data.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
        """
        # Ensure float64 precision
        X_curr = np.array(X, dtype=FLOAT_PRECISION)

        # 1. Variance Thresholding (Sanitization)
        # Removes constant features which can cause instability in downstream scalers
        self.vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_curr = self.vt.fit_transform(X_curr)

        # 2. Yeo-Johnson Power Transformation
        if APPLY_YEO_JOHNSON:
            # standardize=False because we apply StandardScaler explicitly afterwards
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
            X_curr = self.pt.fit_transform(X_curr)

        # 3. Standard Scaling
        if APPLY_SCALING:
            self.scaler = StandardScaler()
            self.scaler.fit(X_curr)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed feature matrix in float64.
        """
        # Ensure float64 precision
        X_curr = np.array(X, dtype=FLOAT_PRECISION)

        # 1. Apply Variance Threshold
        if self.vt is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        X_curr = self.vt.transform(X_curr)

        # 2. Apply Power Transformation
        if APPLY_YEO_JOHNSON:
            if self.pt is None:
                raise RuntimeError("PowerTransformer enabled but not fitted.")
            X_curr = self.pt.transform(X_curr)

        # 3. Apply Standard Scaling
        if APPLY_SCALING:
            if self.scaler is None:
                raise RuntimeError("StandardScaler enabled but not fitted.")
            X_curr = self.scaler.transform(X_curr)

        return X_curr

    def fit_transform(self, X):
        """
        Fits on X and returns the transformed version.
        """
        self.fit(X)
        return self.transform(X)


def get_preprocessed_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the preprocessing pipeline with caching.

    Logic:
    1. Checks if transformed .npy files exist in CACHE_DIR.
    2. If yes and load_cached_data=True, loads and returns them.
    3. If no, fits RobustPreprocessor on X_train, transforms all splits,
       saves to cache, and returns them.

    Args:
        X_train (pd.DataFrame): Raw training features.
        X_val (pd.DataFrame): Raw validation features.
        X_test (pd.DataFrame): Raw test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """
    # Define cache paths
    cache_train_path = os.path.join(CACHE_DIR, "X_train_transformed.npy")
    cache_val_path = os.path.join(CACHE_DIR, "X_val_transformed.npy")
    cache_test_path = os.path.join(CACHE_DIR, "X_test_transformed.npy")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        X_train_trans = np.load(cache_train_path)
        X_val_trans = np.load(cache_val_path)
        X_test_trans = np.load(cache_test_path)
        return X_train_trans, X_val_trans, X_test_trans

    print("Preprocessing data from scratch...")

    # Initialize and fit preprocessor
    preprocessor = RobustPreprocessor()
    preprocessor.fit(X_train)

    # Transform all splits
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save to cache
    print(f"Saving preprocessed data to {CACHE_DIR}...")
    np.save(cache_train_path, X_train_trans)
    np.save(cache_val_path, X_val_trans)
    np.save(cache_test_path, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
