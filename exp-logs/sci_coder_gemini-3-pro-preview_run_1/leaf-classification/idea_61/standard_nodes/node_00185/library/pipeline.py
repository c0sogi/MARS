import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, FLOAT_PRECISION
from library.utils import get_config_hash


class SanitizedTransformer:
    """
    Implements the Sanitized High-Precision Pipeline.

    Pipeline Steps:
    1. Sanitization: VarianceThreshold(threshold=0) to remove constant features.
    2. Transformation: PowerTransformer(method='yeo-johnson', standardize=False).
    3. Scaling: StandardScaler.

    Attributes:
        variance_selector: Removes constant features.
        power_transformer: Applies Yeo-Johnson transformation to stabilize variance.
        scaler: Standardizes features to zero mean and unit variance.
    """

    def __init__(self):
        self.variance_selector = VarianceThreshold(threshold=0.0)
        self.power_transformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the pipeline components sequentially on the input data.

        Args:
            X (pd.DataFrame or np.ndarray): Training data features.
            y: Ignored (unsupervised transformation).

        Returns:
            self: The fitted transformer instance.
        """
        # Ensure input is float64 numpy array
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(FLOAT_PRECISION)
        else:
            X_data = X.astype(FLOAT_PRECISION)

        # 1. Sanitization: Variance Threshold
        # Fit to find non-constant features
        self.variance_selector.fit(X_data)
        X_sanitized = self.variance_selector.transform(X_data)

        # 2. Transformation: Yeo-Johnson
        # Note: We fit on the sanitized data
        self.power_transformer.fit(X_sanitized)
        X_transformed = self.power_transformer.transform(X_sanitized)

        # 3. Scaling: Standard Scaler
        # Fit on the power-transformed data
        self.scaler.fit(X_transformed)

        return self

    def transform(self, X):
        """
        Applies the fitted pipeline to transform input data.

        Args:
            X (pd.DataFrame or np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data matrix in float64 precision.
        """
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(FLOAT_PRECISION)
        else:
            X_data = X.astype(FLOAT_PRECISION)

        # 1. Sanitization
        X_sanitized = self.variance_selector.transform(X_data)

        # 2. Transformation
        X_transformed = self.power_transformer.transform(X_sanitized)

        # 3. Scaling
        X_scaled = self.scaler.transform(X_transformed)

        return X_scaled.astype(FLOAT_PRECISION)


def get_pipeline_hash():
    """
    Generates a unique hash for the current pipeline configuration.
    Used for cache invalidation.
    """
    config = {
        "steps": [
            "VarianceThreshold(threshold=0)",
            "PowerTransformer(method='yeo-johnson', standardize=False)",
            "StandardScaler()",
        ],
        "precision": str(FLOAT_PRECISION),
    }
    return get_config_hash(config)


def run_pipeline(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the pipeline execution for Train, Validation, and Test sets.
    Handles caching of the transformed numpy arrays to disk.

    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        load_cached_data (bool): If True, attempts to load processed data from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Generate config hash to ensure cache validity
    pipeline_hash = get_pipeline_hash()

    # Define cache file paths
    train_cache = os.path.join(WORKING_DIR, f"train_transformed_{pipeline_hash}.npy")
    val_cache = os.path.join(WORKING_DIR, f"val_transformed_{pipeline_hash}.npy")
    test_cache = os.path.join(WORKING_DIR, f"test_transformed_{pipeline_hash}.npy")

    # Check if all required cache files exist
    caches_exist = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    # 1. Try to load from cache
    if load_cached_data and caches_exist:
        print(f"Loading cached transformed data from {WORKING_DIR}...")
        try:
            X_train_trans = np.load(train_cache)
            X_val_trans = np.load(val_cache)
            X_test_trans = np.load(test_cache)

            # Verify precision and shape consistency
            # Cite debug_lesson_14: Validate Cache Consistency Against Runtime Configuration
            if X_train_trans.dtype != FLOAT_PRECISION:
                print("Cache precision mismatch. Recomputing...")
            elif X_train_trans.shape[0] != X_train.shape[0]:
                print(
                    f"Cache mismatch: Expected {X_train.shape[0]} training samples, got {X_train_trans.shape[0]}. Recomputing..."
                )
            elif X_val_trans.shape[0] != X_val.shape[0]:
                print(
                    f"Cache mismatch: Expected {X_val.shape[0]} validation samples, got {X_val_trans.shape[0]}. Recomputing..."
                )
            elif X_test_trans.shape[0] != X_test.shape[0]:
                print(
                    f"Cache mismatch: Expected {X_test.shape[0]} test samples, got {X_test_trans.shape[0]}. Recomputing..."
                )
            else:
                return X_train_trans, X_val_trans, X_test_trans
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Running Sanitized High-Precision Pipeline...")

    # Initialize and Fit on Training Data ONLY
    transformer = SanitizedTransformer()
    transformer.fit(X_train)

    # Transform all datasets
    X_train_trans = transformer.transform(X_train)
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # 3. Save to cache
    try:
        np.save(train_cache, X_train_trans)
        np.save(val_cache, X_val_trans)
        np.save(test_cache, X_test_trans)
        print(f"Saved transformed data to {WORKING_DIR}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return X_train_trans, X_val_trans, X_test_trans
