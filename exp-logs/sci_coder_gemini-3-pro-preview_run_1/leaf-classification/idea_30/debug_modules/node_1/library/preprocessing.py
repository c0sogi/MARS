import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import (
    WORKING_DIR,
    FLOAT_PRECISION,
    USE_YEO_JOHNSON,
    USE_STANDARD_SCALER,
    YEO_JOHNSON_STANDARDIZE,
)


class Float64Preprocessor:
    """
    A wrapper class for preprocessing steps ensuring float64 precision.
    Pipelines:
        1. Yeo-Johnson Power Transformation (optional, usually unstandardized here)
        2. Standard Scaling (optional)
    """

    def __init__(self):
        self.pt = None
        self.scaler = None

        if USE_YEO_JOHNSON:
            # We set standardize=False because we might want to apply StandardScaler separately
            # or not at all, giving us granular control.
            self.pt = PowerTransformer(
                method="yeo-johnson", standardize=YEO_JOHNSON_STANDARDIZE
            )

        if USE_STANDARD_SCALER:
            self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (X_train).
        """
        # Ensure input is float64
        X_64 = X.astype(FLOAT_PRECISION, copy=False)

        current_data = X_64

        if self.pt:
            self.pt.fit(current_data)
            # Transform to feed into next step if needed, though fit() usually suffices
            # for independent transformers. However, if transformers were dependent
            # (e.g. scaler depends on distribution shape), we would transform.
            # Here, Yeo-Johnson changes distribution, so Scaler should be fit on transformed data.
            current_data = self.pt.transform(current_data)

        if self.scaler:
            self.scaler.fit(current_data)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to X.
        """
        X_64 = X.astype(FLOAT_PRECISION, copy=False)
        current_data = X_64

        if self.pt:
            current_data = self.pt.transform(current_data)
            # Enforce float64 after sklearn op (sklearn usually preserves it, but safety first)
            current_data = current_data.astype(FLOAT_PRECISION, copy=False)

        if self.scaler:
            current_data = self.scaler.transform(current_data)
            current_data = current_data.astype(FLOAT_PRECISION, copy=False)

        return current_data


def get_preprocessed_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the preprocessing pipeline with caching.

    Args:
        X_train, X_val, X_test: Input raw feature matrices (numpy arrays).
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans)
    """
    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "X_train_transformed.npy")
    cache_val_path = os.path.join(WORKING_DIR, "X_val_transformed.npy")
    cache_test_path = os.path.join(WORKING_DIR, "X_test_transformed.npy")

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

    print("Computing transformations...")

    # Initialize and fit preprocessor
    preprocessor = Float64Preprocessor()
    preprocessor.fit(X_train)

    # Transform datasets
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save to cache
    print(f"Saving preprocessed data to {WORKING_DIR}...")
    np.save(cache_train_path, X_train_trans)
    np.save(cache_val_path, X_val_trans)
    np.save(cache_test_path, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
