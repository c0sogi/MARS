import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from library.config import FLOAT_PRECISION, WORKING_DIR, RANDOM_SEED
from library.data_loader import load_data


class IterativeGaussianizer(BaseEstimator, TransformerMixin):
    """
    Applies feature-wise Yeo-Johnson transformation followed by Standardization.

    Modified to remove Whitening PCA and iterative stages, as global decorrelation
    destroys class separability for generative classifiers like LDA.
    Cite solution_lesson_node_00144.
    """

    def __init__(self):
        self.pt = None

    def fit(self, X, y=None):
        """
        Fits the transformation pipeline on X.
        """
        # Ensure high precision
        X_curr = X.astype(FLOAT_PRECISION)

        # Apply PowerTransformer with standardization
        # This performs feature-wise Gaussianization and scaling
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)
        self.pt.fit(X_curr)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to X.
        """
        X_curr = X.astype(FLOAT_PRECISION)
        X_curr = self.pt.transform(X_curr)

        return X_curr.astype(FLOAT_PRECISION)


def get_transformed_data(load_cached_data=True):
    """
    Loads raw data, applies the Iterative Gaussianization pipeline, and manages caching.

    Args:
        load_cached_data (bool): If True, attempts to load transformed data from disk.

    Returns:
        tuple: ((X_train, y_train, ids_train),
                (X_val, y_val, ids_val),
                (X_test, ids_test))
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "X_train_transformed.npy")
    cache_val_path = os.path.join(WORKING_DIR, "X_val_transformed.npy")
    cache_test_path = os.path.join(WORKING_DIR, "X_test_transformed.npy")

    # Load raw data (we always need y and ids)
    # Pass the caching flag to the data loader as well
    (
        (X_train_raw, y_train, ids_train),
        (X_val_raw, y_val, ids_val),
        (X_test_raw, ids_test),
    ) = load_data(load_cached_data=load_cached_data)

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            try:
                print("Loading transformed data from cache...")
                X_train = np.load(cache_train_path).astype(FLOAT_PRECISION)
                X_val = np.load(cache_val_path).astype(FLOAT_PRECISION)
                X_test = np.load(cache_test_path).astype(FLOAT_PRECISION)

                return (
                    (X_train, y_train, ids_train),
                    (X_val, y_val, ids_val),
                    (X_test, ids_test),
                )
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")
        else:
            print("Transformed data cache not found. Computing...")
    else:
        print("Ignoring cache. Computing transformed data...")

    # 2. Compute transformations
    print("Fitting Gaussianizer on Training Data...")
    transformer = IterativeGaussianizer()
    transformer.fit(X_train_raw)

    print("Applying transformations...")
    X_train = transformer.transform(X_train_raw)
    X_val = transformer.transform(X_val_raw)
    X_test = transformer.transform(X_test_raw)

    # 3. Save to cache
    try:
        np.save(cache_train_path, X_train)
        np.save(cache_val_path, X_val)
        np.save(cache_test_path, X_test)
        print(f"Saved transformed data to {WORKING_DIR}")
    except Exception as e:
        print(f"Warning: Failed to save transformed data cache: {e}")

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)
