import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import CACHE_DIR
from library.data import get_data


class HighPrecisionTransformer:
    """
    Implements the 'Sanitized Pipeline' for signal conditioning.
    Enforces float64 precision and executes a strict sequence:
    1. VarianceThreshold(threshold=0) (Sanitization)
    2. PowerTransformer(method='yeo-johnson', standardize=False)
    3. StandardScaler
    """

    def __init__(self):
        self.selector = VarianceThreshold(threshold=0)
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fits the transformers on the training data.
        """
        # Enforce float64 precision
        X = np.array(X, dtype=np.float64)

        # 1. Sanitization: Remove constant features
        # This prevents downstream scalers from exploding on null features
        X_sel = self.selector.fit_transform(X)

        # 2. Transformation: Gaussianize
        X_pt = self.pt.fit_transform(X_sel)

        # 3. Scaling: Standardize
        self.scaler.fit(X_pt)

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the fitted transformations to new data.
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        # Enforce float64 precision
        X = np.array(X, dtype=np.float64)

        # 1. Sanitization
        X_sel = self.selector.transform(X)

        # 2. Transformation
        X_pt = self.pt.transform(X_sel)

        # 3. Scaling
        X_scaled = self.scaler.transform(X_pt)

        return X_scaled


def get_preprocessed_data(load_cached_data: bool = True, debug_size: int = None):
    """
    Retrieves and preprocesses the dataset.
    Handles caching of the transformed (expensive) feature matrices.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_size (int): Optional truncation size for debugging.

    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val, X_test, test_ids.
              X arrays are transformed float64 numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Construct cache filenames
    # We include debug_size in the filename to avoid cache pollution
    suffix = "" if debug_size is None else f"_debug_{debug_size}"

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, f"X_train_transformed{suffix}.npy"),
        "X_val": os.path.join(CACHE_DIR, f"X_val_transformed{suffix}.npy"),
        "X_test": os.path.join(CACHE_DIR, f"X_test_transformed{suffix}.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        try:
            # Load transformed features
            X_train = np.load(cache_files["X_train"])
            X_val = np.load(cache_files["X_val"])
            X_test = np.load(cache_files["X_test"])

            # Retrieve labels and IDs from the data module (fast)
            # We pass load_cached_data=True to leverage the raw feature cache
            raw_data = get_data(load_cached_data=True, debug_size=debug_size)

            return {
                "X_train": X_train,
                "y_train": raw_data["y_train"],
                "X_val": X_val,
                "y_val": raw_data["y_val"],
                "X_test": X_test,
                "test_ids": raw_data["test_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print("Preprocessing data from scratch...")

    # 1. Get raw data (features + geometry)
    raw_data = get_data(load_cached_data=True, debug_size=debug_size)

    # 2. Initialize and Fit Transformer
    transformer = HighPrecisionTransformer()

    # Fit ONLY on training data (Inductive)
    print("Fitting transformer on training set...")
    transformer.fit(raw_data["X_train"])

    # 3. Transform all sets
    print("Transforming datasets...")
    X_train_trans = transformer.transform(raw_data["X_train"])
    X_val_trans = transformer.transform(raw_data["X_val"])
    X_test_trans = transformer.transform(raw_data["X_test"])

    # 4. Save to cache
    print("Saving transformed data to cache...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return {
        "X_train": X_train_trans,
        "y_train": raw_data["y_train"],
        "X_val": X_val_trans,
        "y_val": raw_data["y_val"],
        "X_test": X_test_trans,
        "test_ids": raw_data["test_ids"],
    }
