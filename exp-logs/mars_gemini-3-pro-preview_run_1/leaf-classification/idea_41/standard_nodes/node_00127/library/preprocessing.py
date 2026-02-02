import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import CACHE_DIR, PRECISION_TYPE
from library.utils import validate_precision


class HighPrecisionTransformer:
    """
    A wrapper class for the inductive preprocessing pipeline.
    Applies Yeo-Johnson Power Transformation followed by Standard Scaling,
    strictly maintaining float64 precision.
    """

    def __init__(self):
        # As per instructions: Yeo-Johnson with standardize=False
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        # Followed by Standard Scaling
        self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).
        """
        validate_precision(X, "Transformer Fit Input")

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform data to intermediate state to fit StandardScaler
        # We must ensure intermediate result is also float64
        X_pt = self.pt.transform(X).astype(PRECISION_TYPE)

        # Fit StandardScaler on power-transformed data
        self.ss.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.
        """
        validate_precision(X, "Transformer Transform Input")

        # Apply Power Transformation
        X_pt = self.pt.transform(X).astype(PRECISION_TYPE)

        # Apply Standard Scaling
        X_scaled = self.ss.transform(X_pt).astype(PRECISION_TYPE)

        return X_scaled


def preprocess_features(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the HighPrecisionTransformer pipeline to the datasets.
    Implements caching to disk to avoid re-computing transformations.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_transformed, X_val_transformed, X_test_transformed)
    """
    # Define cache paths
    path_train = os.path.join(CACHE_DIR, "X_train_transformed.npy")
    path_val = os.path.join(CACHE_DIR, "X_val_transformed.npy")
    path_test = os.path.join(CACHE_DIR, "X_test_transformed.npy")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if cache exists and is requested
    if (
        load_cached_data
        and os.path.exists(path_train)
        and os.path.exists(path_val)
        and os.path.exists(path_test)
    ):
        print("Loading transformed features from cache...")
        try:
            X_train_trans = np.load(path_train)
            X_val_trans = np.load(path_val)
            X_test_trans = np.load(path_test)

            # Validate precision of loaded data
            validate_precision(X_train_trans, "Cached X_train_transformed")
            validate_precision(X_val_trans, "Cached X_val_transformed")
            validate_precision(X_test_trans, "Cached X_test_transformed")

            return X_train_trans, X_val_trans, X_test_trans
        except Exception as e:
            print(f"Failed to load transformed cache: {e}. Re-computing...")

    print("Computing feature transformations...")

    # Validate inputs
    validate_precision(X_train, "X_train (Input)")
    validate_precision(X_val, "X_val (Input)")
    validate_precision(X_test, "X_test (Input)")

    # Initialize and fit transformer (Inductive: Fit on Train only)
    transformer = HighPrecisionTransformer()
    transformer.fit(X_train)

    # Transform all sets
    X_train_trans = transformer.transform(X_train)
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # Save to cache
    print("Saving transformed features to cache...")
    np.save(path_train, X_train_trans)
    np.save(path_val, X_val_trans)
    np.save(path_test, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
