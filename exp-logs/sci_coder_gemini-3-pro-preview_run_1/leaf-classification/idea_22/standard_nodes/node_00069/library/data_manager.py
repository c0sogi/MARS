import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import METADATA_DIR, WORKING_DIR, FEATURES


class InductivePreprocessor:
    """
    A preprocessing pipeline that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling. It ensures that the transformation parameters
    are learned exclusively from the training data (Inductive Bias).
    """

    def __init__(self):
        # standardize=False for PowerTransformer because we apply StandardScaler afterwards
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the transformers on the training data.
        Args:
            X (np.ndarray): Training features.
        """
        self.pt.fit(X)
        # Transform X temporarily to fit the scaler on the power-transformed space
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Apply the learned transformations to new data.
        Args:
            X (np.ndarray): Features to transform.
        Returns:
            np.ndarray: Transformed features in float64.
        """
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled.astype(np.float64)

    def fit_transform(self, X):
        """
        Fit on X and return the transformed version.
        """
        self.fit(X)
        return self.transform(X)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, applies the inductive preprocessing pipeline, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_train, X_val, X_test: Preprocessed feature arrays (float64).
            - y_train, y_val: Integer-encoded target arrays.
            - test_ids: Array of IDs for the test set.
            - classes: Array of class names corresponding to the integer labels.
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
        "classes": os.path.join(WORKING_DIR, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Loading raw data from metadata...")
    # Load raw CSVs
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Extract Features (Deterministic order from config)
    # Ensure float64 precision
    raw_X_train = df_train[FEATURES].values.astype(np.float64)
    raw_X_val = df_val[FEATURES].values.astype(np.float64)
    raw_X_test = df_test[FEATURES].values.astype(np.float64)

    # Extract IDs and Targets
    test_ids = df_test["id"].values
    y_train_raw = df_train["species"].values
    y_val_raw = df_val["species"].values

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # Apply Inductive Preprocessing
    print("Applying Inductive Preprocessing (Yeo-Johnson + StandardScaler)...")
    preprocessor = InductivePreprocessor()

    # Fit only on Train
    preprocessor.fit(raw_X_train)

    # Transform all sets
    X_train = preprocessor.transform(raw_X_train)
    X_val = preprocessor.transform(raw_X_val)
    X_test = preprocessor.transform(raw_X_test)

    # Save to cache
    print(f"Saving processed data to {WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
