import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR
from library.data_loader import load_datasets


class Preprocessor:
    """
    Encapsulates the inductive preprocessing pipeline.

    Pipeline:
    1. Yeo-Johnson Power Transformation (standardize=False)
    2. Standard Scaling

    Attributes:
        pt (PowerTransformer): The power transformer instance.
        scaler (StandardScaler): The standard scaler instance.
    """

    def __init__(self):
        # method='yeo-johnson' works with positive and negative values
        # standardize=False because we apply StandardScaler explicitly afterwards
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).

        Args:
            X (pd.DataFrame or np.ndarray): Training features.

        Returns:
            self
        """
        # Ensure input is float64
        X_float = np.array(X, dtype=np.float64)

        # Fit PowerTransformer
        self.pt.fit(X_float)

        # Transform to get intermediate state for Scaler fitting
        X_pt = self.pt.transform(X_float)

        # Fit StandardScaler
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features in float64 precision.
        """
        X_float = np.array(X, dtype=np.float64)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_float)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Strictly enforce float64 return type
        return X_scaled.astype(np.float64)


def get_preprocessed_data(load_cached_data=True):
    """
    Orchestrates the loading, preprocessing, and caching of datasets.

    1. Checks for cached transformed data in WORKING_DIR.
    2. If not found or reload forced:
       - Loads raw data via library.data_loader.
       - Fits Preprocessor on Training set ONLY.
       - Transforms Train, Val, and Test sets.
       - Caches the results as .npy files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
               X arrays are np.float64, y and ids are original types.
    """
    # Define cache file paths for transformed data
    cache_paths = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "y_train": os.path.join(
            WORKING_DIR, "y_train_transformed.npy"
        ),  # y doesn't change, but we cache for consistency
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids_transformed.npy"),
        "classes": os.path.join(WORKING_DIR, "classes_transformed.npy"),
    }

    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_paths.values())
        if all_exist:
            print("Loading preprocessed data from cache...")
            X_train = np.load(cache_paths["X_train"])
            y_train = np.load(cache_paths["y_train"], allow_pickle=True)
            X_val = np.load(cache_paths["X_val"])
            y_val = np.load(cache_paths["y_val"], allow_pickle=True)
            X_test = np.load(cache_paths["X_test"])
            test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)
            classes = np.load(cache_paths["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        else:
            print("Preprocessed cache miss. Processing from scratch...")

    # 2. Load raw data (this handles its own caching of the raw split)
    # raw_X are DataFrames with alphanumeric column ordering
    (
        raw_X_train,
        raw_y_train,
        raw_X_val,
        raw_y_val,
        raw_X_test,
        raw_test_ids,
        raw_classes,
    ) = load_datasets(load_cached_data=load_cached_data)

    print("Fitting preprocessor on training data...")
    preprocessor = Preprocessor()
    preprocessor.fit(raw_X_train)

    print("Transforming datasets...")
    X_train_trans = preprocessor.transform(raw_X_train)
    X_val_trans = preprocessor.transform(raw_X_val)
    X_test_trans = preprocessor.transform(raw_X_test)

    # 3. Save to cache
    print("Caching preprocessed data...")
    np.save(cache_paths["X_train"], X_train_trans)
    np.save(cache_paths["y_train"], raw_y_train)

    np.save(cache_paths["X_val"], X_val_trans)
    np.save(cache_paths["y_val"], raw_y_val)

    np.save(cache_paths["X_test"], X_test_trans)
    np.save(cache_paths["test_ids"], raw_test_ids)

    np.save(cache_paths["classes"], raw_classes)

    return (
        X_train_trans,
        raw_y_train,
        X_val_trans,
        raw_y_val,
        X_test_trans,
        raw_test_ids,
        raw_classes,
    )
