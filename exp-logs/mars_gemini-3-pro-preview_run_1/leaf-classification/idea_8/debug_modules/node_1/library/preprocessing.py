import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from library.config import (
    WORKING_DIR,
    POWER_TRANSFORM_METHOD,
)
from library.data_loader import load_data


class FeaturePreprocessor:
    """
    Encapsulates the feature transformation pipeline required for LDA.
    Includes Power Transformation (Yeo-Johnson) to Gaussianize features
    and Standard Scaling for numerical stability.
    """

    def __init__(self):
        # Initialize the pipeline.
        # We set standardize=False in PowerTransformer to allow the StandardScaler
        # to handle the scaling step explicitly, as per the architectural design.
        self.pipeline = Pipeline(
            [
                (
                    "power",
                    PowerTransformer(method=POWER_TRANSFORM_METHOD, standardize=False),
                ),
                ("scaler", StandardScaler()),
            ]
        )

    def fit_transform(self, X):
        """
        Fits the pipeline to the data and returns the transformed version.
        Args:
            X (array-like): Training data.
        Returns:
            array-like: Transformed training data.
        """
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        """
        Applies the fitted pipeline to new data.
        Args:
            X (array-like): Data to transform.
        Returns:
            array-like: Transformed data.
        """
        return self.pipeline.transform(X)


def get_preprocessed_data(load_cached_data=True):
    """
    Retrieves preprocessed data, using local caching to optimize runtime.

    Logic:
    1. Checks if transformed .npy files exist in WORKING_DIR.
    2. If yes and load_cached_data is True, loads and returns them.
    3. If no, loads raw data, computes transformations, saves to cache, and returns.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
               X arrays are transformed (PowerTransform + Scaling).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths for transformed features
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading transformed data from cache...")
            X_train_trans = np.load(cache_files["X_train"])
            X_val_trans = np.load(cache_files["X_val"])
            X_test_trans = np.load(cache_files["X_test"])

            # We need y and ids. We call load_data with caching enabled to get them quickly.
            # We ignore the raw X returned by load_data.
            _, y_train, _, y_val, _, test_ids = load_data(load_cached_data=True)

            return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids

    # If cache miss or forced reload:
    print("Computing feature transformations...")

    # Load raw data
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids = load_data(
        load_cached_data=load_cached_data
    )

    # Initialize and apply preprocessor
    preprocessor = FeaturePreprocessor()

    # Fit on Train, Transform everything
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Save to cache
    print(f"Saving transformed data to {WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids
