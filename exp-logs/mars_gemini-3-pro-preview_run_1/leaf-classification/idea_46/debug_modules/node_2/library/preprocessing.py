import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, FLOAT_PRECISION


class InductivePreprocessor:
    """
    Implements the High-Precision Homoscedasticity Pipeline.
    Wraps Yeo-Johnson PowerTransformer and StandardScaler with strict float64 enforcement.
    """

    def __init__(self):
        # Initialize transformers
        # standardize=False because we apply StandardScaler explicitly afterwards
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X):
        """
        Fits the pipeline parameters on the provided data (Training set).
        """
        # Enforce precision
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(FLOAT_PRECISION)
        else:
            X_data = X.astype(FLOAT_PRECISION)

        # Fit PowerTransformer
        self.pt.fit(X_data)

        # Transform to intermediate state to fit Scaler
        X_pt = self.pt.transform(X_data)

        # Fit StandardScaler
        self.scaler.fit(X_pt)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        # Enforce precision
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(FLOAT_PRECISION)
        else:
            X_data = X.astype(FLOAT_PRECISION)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_data)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Ensure output is strictly float64
        return X_scaled.astype(FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Fits and transforms the data in one step.
        """
        self.fit(X)
        return self.transform(X)


def get_preprocessed_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the preprocessing pipeline with caching.

    Args:
        X_train, X_val, X_test: Input features (DataFrame or numpy array).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Tuple of (X_train_transformed, X_val_transformed, X_test_transformed) as numpy arrays.
    """
    # Define cache paths
    cache_files = {
        "train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        X_train_trans = np.load(cache_files["train"])
        X_val_trans = np.load(cache_files["val"])
        X_test_trans = np.load(cache_files["test"])
        return X_train_trans, X_val_trans, X_test_trans

    print(
        "Cache missing or reload requested. Running High-Precision Homoscedasticity Pipeline..."
    )

    # Initialize Preprocessor
    preprocessor = InductivePreprocessor()

    # Fit ONLY on Training Data
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    # Transform all sets
    print("Transforming datasets...")
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save to cache
    print(f"Saving transformed data to {WORKING_DIR}...")
    np.save(cache_files["train"], X_train_trans)
    np.save(cache_files["val"], X_val_trans)
    np.save(cache_files["test"], X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
