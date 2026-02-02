import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from library.config import CACHE_DIR, FLOAT_PRECISION, SEED
from library.data_loader import load_dataset


class MarginalGaussianizer(BaseEstimator, TransformerMixin):
    """
    Implements a simple Marginal Gaussianization pipeline.
    Cite {solution_lesson_node_00044}: Prefer Marginal Gaussianization over Iterative.
    Cite {solution_lesson_node_00025}: Avoid redundant normalization.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("pt", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("ss", StandardScaler()),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        X_t = self.pipeline.transform(X)
        return X_t.astype(FLOAT_PRECISION)


def get_preprocessed_data(load_cached_data=True):
    """
    Orchestrates data loading, preprocessing (Iterative Gaussianization), and caching.

    Args:
        load_cached_data (bool): Whether to attempt loading transformed data from disk.

    Returns:
        tuple: (train_data, val_data, test_data)
            train_data: (X_train_trans, y_train, ids_train)
            val_data:   (X_val_trans, y_val, ids_val)
            test_data:  (X_test_trans, ids_test)
    """
    # Define cache paths for transformed features
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test_transformed.npy"),
    }

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if we can load from cache
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    # We always need y and ids, so we load the raw dataset structure first.
    # The data_loader has its own caching mechanism for the raw splits.
    (
        (X_train_raw, y_train, ids_train),
        (X_val_raw, y_val, ids_val),
        (X_test_raw, ids_test),
    ) = load_dataset(load_cached_data=load_cached_data)

    if load_cached_data and cache_exists:
        print(f"Loading preprocessed (Gaussianized) data from {CACHE_DIR}...")
        try:
            X_train = np.load(cache_files["X_train"])
            X_val = np.load(cache_files["X_val"])
            X_test = np.load(cache_files["X_test"])

            return (
                (X_train, y_train, ids_train),
                (X_val, y_val, ids_val),
                (X_test, ids_test),
            )
        except Exception as e:
            print(f"Failed to load preprocessed cache: {e}. Re-running pipeline...")

    # If cache miss or force reload, run the pipeline
    print("Fitting Iterative Gaussianization pipeline...")

    # Initialize and fit the transformer
    gaussianizer = MarginalGaussianizer()
    gaussianizer.fit(X_train_raw)

    print("Transforming datasets...")
    X_train = gaussianizer.transform(X_train_raw)
    X_val = gaussianizer.transform(X_val_raw)
    X_test = gaussianizer.transform(X_test_raw)

    # Save to cache
    print(f"Saving preprocessed data to {CACHE_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["X_test"], X_test)

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)
