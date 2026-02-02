import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import IDEA_DIR
from library.data_loader import load_datasets


class RobustPreprocessor:
    """
    Implements the inductive preprocessing pipeline:
    1. PowerTransformer (Yeo-Johnson, standardize=False)
    2. StandardScaler (Fitted on Train only)

    Maintains float64 precision during transformation and casts to float32
    only at the output.
    """

    def __init__(self):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fits the transformers on the provided data (Training set).

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
        """
        # Ensure input is float64
        X_64 = np.array(X, dtype=np.float64)

        # Fit PowerTransformer
        self.pt.fit(X_64)

        # Transform using PT to get data for Scaler fitting
        X_pt = self.pt.transform(X_64)

        # Fit StandardScaler
        self.scaler.fit(X_pt)

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features in float32.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        # Ensure input is float64
        X_64 = np.array(X, dtype=np.float64)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_64)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Cast to float32 as a regularizer and for memory efficiency
        return X_scaled.astype(np.float32)

    def fit_transform(self, X):
        """
        Fits and transforms the data.
        """
        return self.fit(X).transform(X)


def preprocess_data(load_cached_data: bool = True):
    """
    Loads raw data, applies the RobustPreprocessor pipeline, and manages caching
    of the transformed arrays.

    The pipeline is inductive:
    - Fit on Train
    - Transform Train, Val, Test

    Args:
        load_cached_data (bool): If True, attempts to load transformed data from disk.

    Returns:
        tuple: (X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids)
    """
    # Define cache paths for transformed data
    cache_files = {
        "X_train": os.path.join(IDEA_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(IDEA_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(IDEA_DIR, "X_test_transformed.npy"),
        # We reuse the raw targets/ids from data_loader, but for completeness of
        # the API, we will return them. We don't need to cache them again specifically
        # here as data_loader handles them, but we need to fetch them.
    }

    # Ensure working directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        try:
            print(f"Attempting to load transformed data from {IDEA_DIR}...")
            if all(os.path.exists(path) for path in cache_files.values()):
                X_train_trans = np.load(cache_files["X_train"])
                X_val_trans = np.load(cache_files["X_val"])
                X_test_trans = np.load(cache_files["X_test"])

                # We still need y_train, y_val, test_ids.
                # We load them via data_loader (it has its own caching).
                _, y_train, _, y_val, _, test_ids = load_datasets(load_cached_data=True)

                print("Successfully loaded transformed data from cache.")
                return (
                    X_train_trans,
                    y_train,
                    X_val_trans,
                    y_val,
                    X_test_trans,
                    test_ids,
                )
            else:
                print("Transformed data cache incomplete. Processing from scratch...")
        except Exception as e:
            print(f"Error loading transformed cache: {e}. Processing from scratch...")

    # Load raw data (float64 DataFrames)
    print("Loading raw datasets for preprocessing...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids = load_datasets(
        load_cached_data=True
    )

    # Initialize Preprocessor
    print("Initializing RobustPreprocessor (Yeo-Johnson + StandardScaler)...")
    preprocessor = RobustPreprocessor()

    # Fit on Training Data ONLY
    print("Fitting preprocessor on Training set...")
    preprocessor.fit(X_train_raw)

    # Transform all sets
    print("Transforming Training set...")
    X_train_trans = preprocessor.transform(X_train_raw)

    print("Transforming Validation set...")
    X_val_trans = preprocessor.transform(X_val_raw)

    print("Transforming Test set...")
    X_test_trans = preprocessor.transform(X_test_raw)

    # Save to cache
    print(f"Saving transformed data to cache at {IDEA_DIR}...")
    try:
        np.save(cache_files["X_train"], X_train_trans)
        np.save(cache_files["X_val"], X_val_trans)
        np.save(cache_files["X_test"], X_test_trans)
    except Exception as e:
        print(f"Warning: Failed to save transformed cache: {e}")

    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids
