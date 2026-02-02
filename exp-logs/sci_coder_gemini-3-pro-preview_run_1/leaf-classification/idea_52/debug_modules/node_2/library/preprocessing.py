import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config


class HighPrecisionTransformer:
    """
    A wrapper around PowerTransformer and StandardScaler that strictly enforces
    float64 precision for the Integral-Morphological High-Precision pipeline.

    Pipeline:
    1. Cast input to float64.
    2. Yeo-Johnson Power Transform (standardize=False).
    3. Standard Scaler (Mean=0, Std=1).
    """

    def __init__(self):
        # standardize=False is crucial as we apply StandardScaler explicitly afterwards
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the transformer on the provided data.

        Args:
            X (array-like): Training data.
            y (ignored): Target labels (not used).

        Returns:
            self
        """
        # Enforce float64 precision
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform data to fit StandardScaler
        # We must transform because SS needs the distribution output by PT
        X_pt = self.pt.transform(X)

        # Fit StandardScaler
        self.ss.fit(X_pt)

        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.

        Args:
            X (array-like): Data to transform.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Enforce float64 precision
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X)

        # Apply StandardScaler
        X_ss = self.ss.transform(X_pt)

        return X_ss.astype(Config.FLOAT_PRECISION)

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in one step.
        Optimized to avoid redundant transformations.

        Args:
            X (array-like): Training data.
            y (ignored): Target labels.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Enforce float64 precision
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Fit and Transform PT
        X_pt = self.pt.fit_transform(X)

        # Fit and Transform SS
        X_ss = self.ss.fit_transform(X_pt)

        return X_ss.astype(Config.FLOAT_PRECISION)


def preprocess_data(
    X_train, X_val, X_test, cache_name="preprocessed_data", load_cached_data=True
):
    """
    Orchestrates the preprocessing pipeline with caching.
    Fits the HighPrecisionTransformer on X_train ONLY, then transforms X_train, X_val, and X_test.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        cache_name (str): Prefix for cached files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, f"{cache_name}_train.npy")
    val_cache = os.path.join(Config.CACHE_DIR, f"{cache_name}_val.npy")
    test_cache = os.path.join(Config.CACHE_DIR, f"{cache_name}_test.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cached preprocessed data from {Config.CACHE_DIR}...")
            try:
                X_train_trans = np.load(train_cache)
                X_val_trans = np.load(val_cache)
                X_test_trans = np.load(test_cache)

                # Verify precision
                if X_train_trans.dtype != Config.FLOAT_PRECISION:
                    print("Cached data precision mismatch. Recomputing...")
                else:
                    return X_train_trans, X_val_trans, X_test_trans
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print(f"Cache miss for {cache_name}. Computing from scratch...")

    # 2. Compute from Scratch
    print("Fitting HighPrecisionTransformer on Training Data...")
    transformer = HighPrecisionTransformer()

    # Inductive Fit: Fit only on Train, then transform all
    X_train_trans = transformer.fit_transform(X_train)

    print("Transforming Validation and Test Data...")
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # 3. Save to Cache
    print(f"Saving preprocessed data to {Config.CACHE_DIR}...")
    np.save(train_cache, X_train_trans)
    np.save(val_cache, X_val_trans)
    np.save(test_cache, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
