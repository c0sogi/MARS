import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import WORKING_DIR
from library.utils import load_data


class InductivePreprocessor:
    """
    Handles feature transformation with high precision (float64).
    Applies Yeo-Johnson Power Transformation followed by Standard Scaling.
    """

    def __init__(self):
        # Yeo-Johnson is used to stabilize variance.
        # standardize=False because we apply StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).
        """
        # Enforce float64 precision
        X_64 = np.array(X, dtype=np.float64)

        # Fit PowerTransformer
        self.pt.fit(X_64)

        # Transform data to fit StandardScaler on the transformed space
        X_pt = self.pt.transform(X_64)
        self.ss.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.
        """
        # Enforce float64 precision
        X_64 = np.array(X, dtype=np.float64)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_64)

        # Apply StandardScaler
        X_scaled = self.ss.transform(X_pt)
        return X_scaled


def get_preprocessed_data(load_cached_data=True):
    """
    Orchestrates data loading, preprocessing, and caching.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        X_train, y_train, X_val, y_val, X_test, ids_test, label_encoder
    """
    # Ensure cache directory exists
    cache_dir = WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Check if all cache files exist
    all_exist = all(os.path.exists(p) for p in paths.values())

    # 1. Try to load from cache
    if load_cached_data and all_exist:
        print("Loading preprocessed data from cache...")
        try:
            X_train = np.load(paths["X_train"])
            y_train = np.load(paths["y_train"])
            X_val = np.load(paths["X_val"])
            y_val = np.load(paths["y_val"])
            X_test = np.load(paths["X_test"])
            ids_test = np.load(paths["ids_test"], allow_pickle=True)
            classes = np.load(paths["classes"], allow_pickle=True)

            # Reconstruct LabelEncoder
            le = LabelEncoder()
            le.classes_ = classes

            return X_train, y_train, X_val, y_val, X_test, ids_test, le
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            # Fall through to computation logic

    # 2. Compute from scratch
    print("Computing preprocessed data from scratch...")

    # Load raw data using library utils
    # Note: load_data returns X (DataFrame), y (Series), ids (Series)
    X_train_raw, y_train_raw, _ = load_data("train")
    X_val_raw, y_val_raw, _ = load_data("val")
    X_test_raw, _, ids_test_raw = load_data("test")

    # Initialize and fit preprocessor on Training Data only (Inductive)
    preprocessor = InductivePreprocessor()
    preprocessor.fit(X_train_raw.values)

    # Transform all splits
    X_train = preprocessor.transform(X_train_raw.values)
    X_val = preprocessor.transform(X_val_raw.values)
    X_test = preprocessor.transform(X_test_raw.values)

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    # Save to cache
    print(f"Saving preprocessed data to {cache_dir}...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["ids_test"], ids_test_raw.values)
    np.save(paths["classes"], le.classes_)

    return X_train, y_train, X_val, y_val, X_test, ids_test_raw.values, le
