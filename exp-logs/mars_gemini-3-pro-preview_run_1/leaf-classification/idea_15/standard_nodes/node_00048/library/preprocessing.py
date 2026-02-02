import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, SEED
from library.data_loader import load_dataset


class GaussianTransformer:
    """
    A transformer pipeline that enforces Gaussianity on the input features.

    Pipeline:
    1. PowerTransformer (Yeo-Johnson) with standardize=False
    2. StandardScaler (Mean=0, Variance=1)

    Ensures all operations are performed in float64 precision.
    """

    def __init__(self):
        # We explicitly separate standardization to control the pipeline steps exactly as requested
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fit the transformer pipeline on the data.

        Args:
            X (array-like): Training data.
            y (ignored): Target data.

        Returns:
            self
        """
        # Ensure float64
        X = np.array(X, dtype=np.float64)

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform to get intermediate state for fitting scaler
        X_pt = self.pt.transform(X)

        # Fit StandardScaler
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Apply the transformation pipeline.

        Args:
            X (array-like): Data to transform.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Ensure float64
        X = np.array(X, dtype=np.float64)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        return X_scaled.astype(np.float64)

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        """
        return self.fit(X, y).transform(X)


def get_preprocessed_data(load_cached_data=True):
    """
    Loads raw data, applies Gaussian transformation, and caches the result.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, y_train, ids_train,
                X_val_trans, y_val, ids_val,
                X_test_trans, ids_test)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths for transformed features
    # Note: y and ids are managed by data_loader's cache, but we return them for convenience.
    cache_paths = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # Check if we can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_paths.values())
        if all_exist:
            print("Loading transformed data from cache...")
            X_train_trans = np.load(cache_paths["X_train"])
            X_val_trans = np.load(cache_paths["X_val"])
            X_test_trans = np.load(cache_paths["X_test"])

            # Load y and ids using data_loader (it handles its own caching)
            # We pass load_cached_data=True to reuse the raw data cache if available
            _, y_train, ids_train, _, y_val, ids_val, _, ids_test = load_dataset(
                load_cached_data=True
            )

            return (
                X_train_trans,
                y_train,
                ids_train,
                X_val_trans,
                y_val,
                ids_val,
                X_test_trans,
                ids_test,
            )
        else:
            print("Transformed data cache incomplete. Processing from scratch...")
    else:
        print("Ignoring transformed data cache. Processing from scratch...")

    # Load raw data
    print("Loading raw data for preprocessing...")
    X_train_raw, y_train, ids_train, X_val_raw, y_val, ids_val, X_test_raw, ids_test = (
        load_dataset(load_cached_data=True)
    )

    # Initialize Transformer
    print("Fitting GaussianTransformer...")
    transformer = GaussianTransformer()

    # Fit on Train only
    transformer.fit(X_train_raw)

    # Transform all sets
    print("Transforming datasets...")
    X_train_trans = transformer.transform(X_train_raw)
    X_val_trans = transformer.transform(X_val_raw)
    X_test_trans = transformer.transform(X_test_raw)

    # Save to cache
    print("Saving transformed data to cache...")
    np.save(cache_paths["X_train"], X_train_trans)
    np.save(cache_paths["X_val"], X_val_trans)
    np.save(cache_paths["X_test"], X_test_trans)

    return (
        X_train_trans,
        y_train,
        ids_train,
        X_val_trans,
        y_val,
        ids_val,
        X_test_trans,
        ids_test,
    )
