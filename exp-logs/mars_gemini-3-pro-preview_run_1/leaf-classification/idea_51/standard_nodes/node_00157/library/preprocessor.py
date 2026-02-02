import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import (
    WORKING_DIR,
    FLOAT_PRECISION,
    SEED,
    USE_YEO_JOHNSON,
    USE_STANDARD_SCALER,
)
from library.data_manager import get_train_data, get_val_data, get_test_data

# Ensure reproducibility
np.random.seed(SEED)


class HighPrecisionPipeline:
    """
    A wrapper class for the inductive transformation pipeline.
    Ensures operations are performed in double precision (float64).
    Fits only on the training set.
    """

    def __init__(self):
        self.pt = None
        self.scaler = None

        # Initialize transformers based on config
        if USE_YEO_JOHNSON:
            # standardize=False because we apply StandardScaler explicitly afterwards
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)

        if USE_STANDARD_SCALER:
            self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the pipeline to the data.
        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
        """
        # Enforce precision
        X_curr = X.astype(FLOAT_PRECISION)

        if self.pt:
            self.pt.fit(X_curr)
            # Transform to get the distribution ready for the scaler
            X_curr = self.pt.transform(X_curr)

        if self.scaler:
            self.scaler.fit(X_curr)

        return self

    def transform(self, X):
        """
        Apply the transformation to the data.
        Args:
            X (np.ndarray): Data to transform.
        Returns:
            np.ndarray: Transformed data in float64.
        """
        X_curr = X.astype(FLOAT_PRECISION)

        if self.pt:
            X_curr = self.pt.transform(X_curr)

        if self.scaler:
            X_curr = self.scaler.transform(X_curr)

        return X_curr.astype(FLOAT_PRECISION)


def get_transformed_data(load_cached_data=True):
    """
    Loads raw data, fits the pipeline on training data, transforms all sets,
    and returns the processed numpy arrays. Implements caching for transformed matrices.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, ids_test)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_X_train = os.path.join(WORKING_DIR, "X_train_transformed.npy")
    cache_X_val = os.path.join(WORKING_DIR, "X_val_transformed.npy")
    cache_X_test = os.path.join(WORKING_DIR, "X_test_transformed.npy")

    # Load raw data (this handles its own caching of the merge step)
    # We need raw y and ids regardless of X caching
    print("Loading raw merged data...")
    df_X_train, y_train, ids_train = get_train_data(load_cached_data)
    df_X_val, y_val, ids_val = get_val_data(load_cached_data)
    df_X_test, _, ids_test = get_test_data(load_cached_data)

    # Check if we can use cached transformed X
    if (
        load_cached_data
        and os.path.exists(cache_X_train)
        and os.path.exists(cache_X_val)
        and os.path.exists(cache_X_test)
    ):

        try:
            print(f"Loading transformed data from {WORKING_DIR}...")
            X_train = np.load(cache_X_train)
            X_val = np.load(cache_X_val)
            X_test = np.load(cache_X_test)

            # Validate shapes against raw data to ensure consistency
            if (
                (X_train.shape[0] == len(y_train))
                and (X_val.shape[0] == len(y_val))
                and (X_test.shape[0] == len(ids_test))
            ):
                return X_train, y_train, X_val, y_val, X_test, ids_test
            else:
                print(
                    "Cached transformed data dimensions do not match raw data. Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load transformed cache: {e}. Recomputing...")

    # If we are here, we need to compute
    print("Fitting HighPrecisionPipeline and transforming data...")

    # Convert DataFrames to numpy arrays with strict precision
    X_train_raw = df_X_train.values.astype(FLOAT_PRECISION)
    X_val_raw = df_X_val.values.astype(FLOAT_PRECISION)
    X_test_raw = df_X_test.values.astype(FLOAT_PRECISION)

    # Initialize and fit pipeline
    pipeline = HighPrecisionPipeline()
    pipeline.fit(X_train_raw)

    # Transform
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # Save to cache
    print(f"Saving transformed data to {WORKING_DIR}...")
    np.save(cache_X_train, X_train)
    np.save(cache_X_val, X_val)
    np.save(cache_X_test, X_test)

    return X_train, y_train, X_val, y_val, X_test, ids_test
