import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from library.config import CACHE_DIR, FLOAT_PRECISION, SEED
from library.utils import set_seed
from library.feature_extraction import load_data_with_features

# Set global seed
set_seed(SEED)


class SanitizedPreprocessor:
    """
    Implements the Sanitized Preprocessing pipeline.

    Sequence:
    1. VarianceThreshold(threshold=0): Removes constant features to prevent numerical explosion.
    2. PowerTransformer(method='yeo-johnson', standardize=False): Stabilizes variance.
    3. StandardScaler: Centers and scales the data.

    Enforces float64 precision throughout to support the High-Precision OAS Discriminant.
    """

    def __init__(self):
        self.variance_selector = VarianceThreshold(threshold=0.0)
        self.power_transformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Fits the pipeline on the training data.
        """
        # Enforce high precision
        X_arr = np.array(X, dtype=FLOAT_PRECISION)

        # 1. Sanitization: Remove constant features
        self.variance_selector.fit(X_arr)
        X_sel = self.variance_selector.transform(X_arr)

        # 2. Stabilization: Power Transform
        self.power_transformer.fit(X_sel)
        X_pt = self.power_transformer.transform(X_sel)

        # 3. Normalization: Standard Scaler
        self.scaler.fit(X_pt)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.
        """
        if not self._is_fitted:
            raise RuntimeError("SanitizedPreprocessor has not been fitted yet.")

        # Enforce high precision
        X_arr = np.array(X, dtype=FLOAT_PRECISION)

        # Apply steps in order using fitted parameters
        X_sel = self.variance_selector.transform(X_arr)
        X_pt = self.power_transformer.transform(X_sel)
        X_scaled = self.scaler.transform(X_pt)

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in one step.
        """
        return self.fit(X, y).transform(X)


def get_preprocessed_data(load_cached_data=True, limit=None):
    """
    Loads raw features, applies the SanitizedPreprocessor, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed .npy files.
        limit (int, optional): Limits the number of samples for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids) as numpy arrays.
    """
    # Define cache filenames based on limit
    suffix = "" if limit is None else f"_limit_{limit}"

    filenames = {
        "X_train": f"X_train_processed{suffix}.npy",
        "y_train": f"y_train_processed{suffix}.npy",
        "X_val": f"X_val_processed{suffix}.npy",
        "y_val": f"y_val_processed{suffix}.npy",
        "X_test": f"X_test_processed{suffix}.npy",
        "test_ids": f"test_ids_processed{suffix}.npy",
    }

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(os.path.join(CACHE_DIR, fname)) for fname in filenames.values()
    )

    if load_cached_data and cache_exists:
        print("[Pipeline] Loading preprocessed data from cache...")
        data = {}
        for key, fname in filenames.items():
            path = os.path.join(CACHE_DIR, fname)
            # allow_pickle=True is needed for string arrays (y_train/val) and potentially object arrays
            data[key] = np.load(path, allow_pickle=True)

        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["test_ids"],
        )

    # If cache miss or force reload, compute from scratch
    print("[Pipeline] Computing preprocessed data from scratch...")

    # 1. Load Raw Features (Hybrid Geometric Fusion)
    # This step uses feature_extraction.py which handles its own parquet caching
    print("[Pipeline] Loading raw training data...")
    X_train_raw, y_train, _ = load_data_with_features(
        "train", load_cached_data=load_cached_data, limit=limit
    )

    print("[Pipeline] Loading raw validation data...")
    X_val_raw, y_val, _ = load_data_with_features(
        "val", load_cached_data=load_cached_data, limit=limit
    )

    print("[Pipeline] Loading raw test data...")
    X_test_raw, _, test_ids = load_data_with_features(
        "test", load_cached_data=load_cached_data, limit=limit
    )

    # 2. Fit Pipeline on Training Data ONLY
    print("[Pipeline] Fitting SanitizedPreprocessor on training data...")
    preprocessor = SanitizedPreprocessor()
    preprocessor.fit(X_train_raw)

    # 3. Transform all splits
    print("[Pipeline] Transforming datasets...")
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 4. Save to Cache
    print(f"[Pipeline] Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(os.path.join(CACHE_DIR, filenames["X_train"]), X_train)
    np.save(os.path.join(CACHE_DIR, filenames["y_train"]), y_train)
    np.save(os.path.join(CACHE_DIR, filenames["X_val"]), X_val)
    np.save(os.path.join(CACHE_DIR, filenames["y_val"]), y_val)
    np.save(os.path.join(CACHE_DIR, filenames["X_test"]), X_test)
    np.save(os.path.join(CACHE_DIR, filenames["test_ids"]), test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids
