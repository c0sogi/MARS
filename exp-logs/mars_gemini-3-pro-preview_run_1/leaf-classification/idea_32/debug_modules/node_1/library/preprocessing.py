import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config


class RobustPipeline:
    """
    A preprocessing pipeline that applies Yeo-Johnson transformation followed by
    Standard Scaling, strictly maintaining float64 precision.

    This pipeline is designed to improve the normality of feature distributions
    while preserving the high numerical precision required for exact linear solvers.
    """

    def __init__(self):
        # Initialize PowerTransformer with Yeo-Johnson method.
        # standardize=False is used because we apply a dedicated StandardScaler afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data.

        Args:
            X (array-like): Training data.

        Returns:
            self
        """
        # Ensure input is float64 to preserve precision during internal calculations
        X_64 = np.array(X, dtype=np.float64)

        # Fit PowerTransformer
        self.pt.fit(X_64)

        # Transform data to intermediate state to fit the Scaler
        X_pt = self.pt.transform(X_64)

        # Fit StandardScaler on the power-transformed data
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (array-like): Data to transform.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Ensure input is float64
        X_64 = np.array(X, dtype=np.float64)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_64)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Explicitly cast to float64 to guarantee output precision
        return X_scaled.astype(np.float64)


def preprocess_datasets(X_train, X_val, X_test, load_cached_data=True):
    """
    Orchestrates the preprocessing of train, validation, and test sets.
    Handles caching of transformed arrays to disk using .npy format.

    The pipeline is fitted ONLY on X_train, then applied to all sets.

    Args:
        X_train, X_val, X_test: Input features (pandas DataFrame or numpy array).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """

    # Define cache file paths
    train_cache_path = os.path.join(config.WORKING_DIR, "X_train_transformed.npy")
    val_cache_path = os.path.join(config.WORKING_DIR, "X_val_transformed.npy")
    test_cache_path = os.path.join(config.WORKING_DIR, "X_test_transformed.npy")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        print("Loading transformed datasets from cache...")
        try:
            X_train_trans = np.load(train_cache_path)
            X_val_trans = np.load(val_cache_path)
            X_test_trans = np.load(test_cache_path)
            return X_train_trans, X_val_trans, X_test_trans
        except Exception as e:
            print(f"Error loading cache: {e}. Re-running preprocessing.")

    # If cache miss or error, run the pipeline
    print("Fitting and applying preprocessing pipeline...")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Initialize pipeline
    pipeline = RobustPipeline()

    # Fit ONLY on training data
    pipeline.fit(X_train)

    # Transform all datasets
    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Save to cache
    np.save(train_cache_path, X_train_trans)
    np.save(val_cache_path, X_val_trans)
    np.save(test_cache_path, X_test_trans)

    print(f"Transformed datasets saved to {config.WORKING_DIR}")

    return X_train_trans, X_val_trans, X_test_trans
