import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config


class HighPrecisionTransformer:
    """
    A pipeline wrapper that applies Yeo-Johnson transformation followed by
    Standard Scaling, strictly maintaining float64 precision.

    Attributes:
        pt (PowerTransformer): Sklearn PowerTransformer with method='yeo-johnson'.
        ss (StandardScaler): Sklearn StandardScaler.
    """

    def __init__(self):
        # standardize=False because we apply explicit StandardScaler afterwards
        # This separates the normality adjustment from the scaling
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the training data.

        Args:
            X (array-like): Training features.

        Returns:
            self
        """
        # Ensure float64 input
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(config.FLOAT_PRECISION)

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform to get intermediate state for StandardScaler fitting
        X_pt = self.pt.transform(X)

        # Fit StandardScaler on the power-transformed data
        self.ss.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.

        Args:
            X (array-like): Features to transform.

        Returns:
            np.ndarray: Transformed features in float64.
        """
        # Ensure float64 input
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(config.FLOAT_PRECISION)

        # Apply Power Transformation
        X_pt = self.pt.transform(X)

        # Apply Standard Scaling
        X_scaled = self.ss.transform(X_pt)

        # Explicit cast to ensure return type is float64
        return X_scaled.astype(config.FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Fits and transforms the data.

        Args:
            X (array-like): Training features.

        Returns:
            np.ndarray: Transformed features in float64.
        """
        self.fit(X)
        return self.transform(X)


def get_transformed_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Manages the caching and transformation of the dataset.

    Checks if transformed data exists in the cache directory. If so, loads it.
    Otherwise, fits the HighPrecisionTransformer on X_train, transforms all sets,
    and saves them to the cache.

    Args:
        X_train (pd.DataFrame): Raw training features (float64).
        X_val (pd.DataFrame): Raw validation features (float64).
        X_test (pd.DataFrame): Raw test features (float64).
        load_cached_data (bool): Whether to attempt loading from the local cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths for transformed data
    cache_paths = {
        "X_train": os.path.join(config.WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(config.WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(config.WORKING_DIR, "X_test_transformed.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_paths.values()):
            print("Loading transformed features from cache...")
            try:
                X_train_trans = np.load(cache_paths["X_train"])
                X_val_trans = np.load(cache_paths["X_val"])
                X_test_trans = np.load(cache_paths["X_test"])
                return X_train_trans, X_val_trans, X_test_trans
            except Exception as e:
                print(f"Failed to load transformed cache: {e}. Re-computing...")
        else:
            print(
                "Transformed cache missing or incomplete. Computing transformations..."
            )

    # Compute transformations
    print("Fitting HighPrecisionTransformer on training data...")
    transformer = HighPrecisionTransformer()

    # Fit only on train, then transform all sets
    # This ensures no data leakage from val/test into the transformation parameters
    X_train_trans = transformer.fit_transform(X_train)

    print("Transforming validation and test data...")
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # Save to cache for future runs
    print(f"Saving transformed features to {config.WORKING_DIR}...")
    np.save(cache_paths["X_train"], X_train_trans)
    np.save(cache_paths["X_val"], X_val_trans)
    np.save(cache_paths["X_test"], X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
