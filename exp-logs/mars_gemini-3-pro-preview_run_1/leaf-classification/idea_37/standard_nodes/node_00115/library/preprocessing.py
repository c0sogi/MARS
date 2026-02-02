import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from library import config


class HighPrecisionPipeline:
    """
    A wrapper class for the inductive transformation pipeline.
    Applies VarianceThreshold, Yeo-Johnson transformation, and Standard Scaling.
    Ensures all operations are performed in float64 precision to support
    exact analytical inference in downstream models.
    """

    def __init__(self):
        # Remove constant features to prevent 0-variance outputs in StandardScaler
        self.selector = VarianceThreshold(threshold=0)
        # Yeo-Johnson is chosen to stabilize variance (heteroscedasticity),
        # particularly for the new geometric features (Aspect Ratio, Solidity).
        # standardize=False allows us to apply StandardScaler explicitly later.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the training data.

        Args:
            X (np.ndarray): Training data matrix (float64).

        Returns:
            self
        """
        # 1. Remove constant features
        self.selector.fit(X)
        X_sel = self.selector.transform(X)

        # 2. Fit PowerTransformer to stabilize variance
        self.pt.fit(X_sel)
        X_pt = self.pt.transform(X_sel)

        # 3. Transform X to fit StandardScaler on the stabilized data
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the fitted transformations to the data.

        Args:
            X (np.ndarray): Data matrix to transform (float64).

        Returns:
            np.ndarray: Transformed data matrix (float64).
        """
        # Apply VarianceThreshold
        X_sel = self.selector.transform(X)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_sel)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Ensure strict float64 return type for numerical stability
        return X_scaled.astype(config.FLOAT_PRECISION)


def process_and_cache_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the preprocessing pipeline with caching.
    Converts inputs to numpy float64, fits the pipeline on training data,
    transforms all datasets, and caches the results.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as numpy arrays.
    """
    # Define cache filenames
    cache_files = {
        "train": config.get_cache_path("X_train_transformed.npy"),
        "val": config.get_cache_path("X_val_transformed.npy"),
        "test": config.get_cache_path("X_test_transformed.npy"),
    }

    # 1. Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(f) for f in cache_files.values()):
            print(f"Loading transformed data from {config.CACHE_DIR}...")
            try:
                X_train_trans = np.load(cache_files["train"])
                X_val_trans = np.load(cache_files["val"])
                X_test_trans = np.load(cache_files["test"])
                return X_train_trans, X_val_trans, X_test_trans
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")
        else:
            print("Transformed cache incomplete or missing. Recomputing...")

    # 2. Prepare Data
    print("Converting inputs to high-precision numpy arrays...")
    # Ensure inputs are float64 before any processing
    X_train_np = X_train.to_numpy(dtype=config.FLOAT_PRECISION)
    X_val_np = X_val.to_numpy(dtype=config.FLOAT_PRECISION)
    X_test_np = X_test.to_numpy(dtype=config.FLOAT_PRECISION)

    # 3. Fit Pipeline
    print("Fitting High-Precision Pipeline on Training Data...")
    pipeline = HighPrecisionPipeline()
    pipeline.fit(X_train_np)

    # 4. Transform Data
    print("Transforming datasets...")
    X_train_trans = pipeline.transform(X_train_np)
    X_val_trans = pipeline.transform(X_val_np)
    X_test_trans = pipeline.transform(X_test_np)

    # 5. Save to Cache
    print(f"Saving transformed data to {config.CACHE_DIR}...")
    try:
        # Ensure directory exists
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        np.save(cache_files["train"], X_train_trans)
        np.save(cache_files["val"], X_val_trans)
        np.save(cache_files["test"], X_test_trans)
        print("Caching complete.")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return X_train_trans, X_val_trans, X_test_trans
