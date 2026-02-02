import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR
from library.data_manager import get_data


class PrecisionPipeline:
    """
    A preprocessing pipeline that applies Yeo-Johnson transformation and Standard Scaling.
    It enforces an inductive fit (fitting only on training data) and quantizes the
    output to float32 to enforce precision consistency.
    """

    def __init__(self):
        # Yeo-Johnson transformation to stabilize variance and make data more Gaussian-like.
        # standardize=False because we apply StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the transformers on the provided data (expected to be training data).
        Args:
            X (np.ndarray): Input data, expected to be float64.
        """
        self.pt.fit(X)
        # Transform temporarily to fit the scaler on the transformed space
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted parameters and casts to float32.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Transformed data in float32.
        """
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        # Quantization: Cast to float32 to filter high-frequency numerical noise
        return X_scaled.astype(np.float32)

    def fit_transform(self, X):
        """
        Fits and transforms the data.
        """
        self.fit(X)
        return self.transform(X)


def get_preprocessed_data(load_cached_data=True):
    """
    Retrieves preprocessed data, utilizing caching to save runtime.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, class_names)
               Feature arrays are float32.
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if cache exists and is valid
    if load_cached_data and all(os.path.exists(path) for path in cache_files.values()):
        print("Loading preprocessed data from cache...")
        X_train_trans = np.load(cache_files["X_train"])
        X_val_trans = np.load(cache_files["X_val"])
        X_test_trans = np.load(cache_files["X_test"])

        # Load the non-transformed components (labels, ids) from the data manager
        # The data manager handles its own caching, so this is efficient.
        _, y_train, _, y_val, _, test_ids, class_names = get_data(load_cached_data=True)

        return (
            X_train_trans,
            y_train,
            X_val_trans,
            y_val,
            X_test_trans,
            test_ids,
            class_names,
        )

    print("Preprocessing data from scratch...")

    # Load raw data (float64)
    X_train, y_train, X_val, y_val, X_test, test_ids, class_names = get_data(
        load_cached_data=True
    )

    # Initialize the pipeline
    pipeline = PrecisionPipeline()

    print("Fitting PrecisionPipeline on Training set...")
    # Fit only on Train, transform Train
    X_train_trans = pipeline.fit_transform(X_train)

    print("Transforming Validation and Test sets...")
    # Transform Val and Test using Train statistics (Inductive)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Save to cache
    print("Saving preprocessed data to cache...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return (
        X_train_trans,
        y_train,
        X_val_trans,
        y_val,
        X_test_trans,
        test_ids,
        class_names,
    )
