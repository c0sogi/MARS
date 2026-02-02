import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import CACHE_DIR, FLOAT_PRECISION
from library.data_loader import load_data, get_features_and_targets


class Float64Preprocessor:
    """
    A preprocessing pipeline that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling, strictly maintaining float64 precision.
    """

    def __init__(self):
        # Yeo-Johnson transformation to stabilize variance and make data more Gaussian-like.
        # standardize=False because we apply a separate StandardScaler afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the training data.

        Args:
            X (np.ndarray): Training features (float64).

        Returns:
            self
        """
        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform data to fit the StandardScaler on the transformed space
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (np.ndarray): Features to transform (float64).

        Returns:
            np.ndarray: Transformed features in float64.
        """
        # Apply PowerTransformer
        X_pt = self.pt.transform(X)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Ensure output is explicitly float64
        return X_scaled.astype(FLOAT_PRECISION)


def get_preprocessed_data(load_cached_data=True):
    """
    Loads raw data, applies the inductive preprocessing pipeline, and manages caching.

    The pipeline is fitted ONLY on the training set, then applied to train, val, and test.
    Results are cached as .npy files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load processed arrays from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, ids_test)
               X arrays are float64, y are string labels, ids are integers.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train_transformed.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val_transformed.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test_transformed.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        try:
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            ids_test = np.load(cache_files["ids_test"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, ids_test
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing preprocessing pipeline from scratch...")

    # Load raw data (DataFrames)
    # Pass load_cached_data=True to allow data_loader to use its own parquet cache
    df_train, df_val, df_test = load_data(load_cached_data=load_cached_data)

    # Extract features and targets/ids
    # get_features_and_targets handles alphanumeric sorting and float64 casting
    X_train_raw, y_train = get_features_and_targets(df_train, is_test=False)
    X_val_raw, y_val = get_features_and_targets(df_val, is_test=False)
    X_test_raw, ids_test = get_features_and_targets(df_test, is_test=True)

    # Initialize and fit preprocessor (Inductive: Fit on Train only)
    preprocessor = Float64Preprocessor()
    preprocessor.fit(X_train_raw)

    # Transform all sets
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Save to cache
    print(f"Saving preprocessed data to {CACHE_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)

    return X_train, y_train, X_val, y_val, X_test, ids_test
