import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, NUMERIC_TYPE
from library.data_loader import load_dataset


class RobustPreprocessor:
    """
    Implements the inductive preprocessing pipeline with Yeo-Johnson Power Transformation
    and Standard Scaling, operating in high precision (float64).
    """

    def __init__(self):
        # standardize=False because we apply StandardScaler explicitly afterwards
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
        """
        # Ensure input is float64
        X = np.array(X, dtype=NUMERIC_TYPE)

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform to feed into StandardScaler fit
        X_pt = self.pt.transform(X)

        # Fit StandardScaler
        self.ss.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features in float64.
        """
        X = np.array(X, dtype=NUMERIC_TYPE)
        X_pt = self.pt.transform(X)
        X_ss = self.ss.transform(X_pt)
        return X_ss.astype(NUMERIC_TYPE)


def get_preprocessed_data(load_cached_data=True):
    """
    Orchestrates the loading, fitting, transforming, and caching of the datasets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: ((X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test))
            X arrays are transformed numpy arrays (float64).
            y arrays are target labels.
            ids arrays are image identifiers.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths for the transformed feature matrices
    path_X_train = os.path.join(WORKING_DIR, "X_train_transformed.npy")
    path_X_val = os.path.join(WORKING_DIR, "X_val_transformed.npy")
    path_X_test = os.path.join(WORKING_DIR, "X_test_transformed.npy")

    # Check cache availability
    cache_exists = (
        os.path.exists(path_X_train)
        and os.path.exists(path_X_val)
        and os.path.exists(path_X_test)
    )

    if load_cached_data and cache_exists:
        print(f"Loading preprocessed data from cache at {WORKING_DIR}...")
        X_train = np.load(path_X_train)
        X_val = np.load(path_X_val)
        X_test = np.load(path_X_test)

        # Load targets and IDs using data_loader (it handles its own caching for these)
        # We discard the raw X returned by load_dataset as we have the transformed versions
        _, y_train, ids_train = load_dataset("train", load_cached_data=True)
        _, y_val, ids_val = load_dataset("val", load_cached_data=True)
        _, _, ids_test = load_dataset("test", load_cached_data=True)

        return (
            (X_train, y_train, ids_train),
            (X_val, y_val, ids_val),
            (X_test, ids_test),
        )

    # If cache miss or forced reload
    print("Processing data from scratch (Fitting Preprocessor)...")

    # 1. Load Raw Data
    # We use load_cached_data=True here to utilize the raw data cache if available
    X_train_raw, y_train, ids_train = load_dataset("train", load_cached_data=True)
    X_val_raw, y_val, ids_val = load_dataset("val", load_cached_data=True)
    X_test_raw, _, ids_test = load_dataset("test", load_cached_data=True)

    # 2. Inductive Fit (Train only)
    preprocessor = RobustPreprocessor()
    print("Fitting RobustPreprocessor on Training data...")
    preprocessor.fit(X_train_raw)

    # 3. Transform all splits
    print("Transforming datasets...")
    X_train_trans = preprocessor.transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # 4. Save to Cache
    print(f"Saving preprocessed data to {WORKING_DIR}...")
    np.save(path_X_train, X_train_trans)
    np.save(path_X_val, X_val_trans)
    np.save(path_X_test, X_test_trans)

    return (
        (X_train_trans, y_train, ids_train),
        (X_val_trans, y_val, ids_val),
        (X_test_trans, ids_test),
    )
