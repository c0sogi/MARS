import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from library import config, data_loader


class Float64Pipeline:
    """
    A wrapper around sklearn transformers to enforce float64 precision
    throughout the pipeline, ensuring high-fidelity parameter estimation
    for the Dual-Precision Gaussian-OAS Discriminant.
    """

    def __init__(self):
        # Initialize the pipeline with Yeo-Johnson (standardize=False) and StandardScaler
        # as specified in the idea description.
        self.pipeline = Pipeline(
            [
                (
                    "yeo_johnson",
                    PowerTransformer(method="yeo-johnson", standardize=False),
                ),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, X):
        """
        Fit the pipeline on the data.
        Args:
            X: Input data (array-like).
        """
        # Enforce float64 precision for parameter estimation
        X_64 = np.array(X, dtype=np.float64)
        self.pipeline.fit(X_64)
        return self

    def transform(self, X):
        """
        Transform the data using the fitted pipeline.
        Args:
            X: Input data (array-like).
        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Enforce float64 input
        X_64 = np.array(X, dtype=np.float64)
        # Transform and ensure output is explicitly float64
        X_transformed = self.pipeline.transform(X_64)
        return X_transformed.astype(np.float64)


def get_preprocessed_data(load_cached_data=True):
    """
    Loads raw data, applies the high-precision inductive preprocessing pipeline,
    and manages caching of the transformed numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            - X_train, X_val, X_test: Transformed features (np.float64)
            - y_train, y_val: Labels
            - test_ids: Image IDs for submission
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache file paths for transformed data
    cache_paths = {
        "X_train": os.path.join(config.CACHE_DIR, "X_train_transformed.npy"),
        "y_train": os.path.join(
            config.CACHE_DIR, "y_train_transformed.npy"
        ),  # Labels don't change but caching ensures consistency
        "X_val": os.path.join(config.CACHE_DIR, "X_val_transformed.npy"),
        "y_val": os.path.join(config.CACHE_DIR, "y_val_transformed.npy"),
        "X_test": os.path.join(config.CACHE_DIR, "X_test_transformed.npy"),
        "test_ids": os.path.join(config.CACHE_DIR, "test_ids_transformed.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_paths.values())

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        X_train = np.load(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"], allow_pickle=True)
        X_val = np.load(cache_paths["X_val"])
        y_val = np.load(cache_paths["y_val"], allow_pickle=True)
        X_test = np.load(cache_paths["X_test"])
        test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Generating preprocessed data from scratch...")

    # 1. Load Raw Data (float64)
    # We pass load_cached_data to the loader as well to utilize its parquet cache if available
    raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, test_ids = (
        data_loader.load_datasets(load_cached_data=load_cached_data)
    )

    # 2. Initialize Pipeline
    pipeline = Float64Pipeline()

    # 3. Inductive Fitting
    # Fit ONLY on the training set (N=712) as per "Inductive Preprocessing Pipeline" requirements.
    # We explicitly reject fitting on test or validation data.
    print("Fitting pipeline on training set...")
    pipeline.fit(raw_X_train)

    # 4. Transform all splits
    print("Transforming datasets...")
    X_train_trans = pipeline.transform(raw_X_train)
    X_val_trans = pipeline.transform(raw_X_val)
    X_test_trans = pipeline.transform(raw_X_test)

    # 5. Cache Results
    print(f"Saving preprocessed data to {config.CACHE_DIR}...")
    np.save(cache_paths["X_train"], X_train_trans)
    np.save(cache_paths["y_train"], raw_y_train)
    np.save(cache_paths["X_val"], X_val_trans)
    np.save(cache_paths["y_val"], raw_y_val)
    np.save(cache_paths["X_test"], X_test_trans)
    np.save(cache_paths["test_ids"], test_ids)

    return X_train_trans, raw_y_train, X_val_trans, raw_y_val, X_test_trans, test_ids
