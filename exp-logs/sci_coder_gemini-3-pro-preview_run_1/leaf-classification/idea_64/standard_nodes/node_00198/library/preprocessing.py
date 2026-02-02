import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, VARIANCE_THRESHOLD, USE_FLOAT64


class RobustPreprocessor:
    """
    Implements the Pipeline Sanitization and Inductive Preprocessing logic.

    Pipeline Steps:
    1. VarianceThreshold: Removes constant/quasi-constant features.
    2. PowerTransformer (Yeo-Johnson): Stabilizes variance and minimizes skewness.
    3. StandardScaler: Centers and scales data to unit variance.

    Attributes:
        pipeline (sklearn.pipeline.Pipeline): The underlying transformation pipeline.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("selector", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
                (
                    "transformer",
                    PowerTransformer(method="yeo-johnson", standardize=False),
                ),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, X):
        """
        Fits the pipeline on the training data.

        Args:
            X (np.ndarray): Training features.

        Returns:
            self
        """
        if USE_FLOAT64:
            X = X.astype(np.float64)

        self.pipeline.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.

        Args:
            X (np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features.
        """
        if USE_FLOAT64:
            X = X.astype(np.float64)

        return self.pipeline.transform(X)

    def fit_transform(self, X):
        """
        Fits the pipeline and transforms the training data in one step.

        Args:
            X (np.ndarray): Training features.

        Returns:
            np.ndarray: Transformed training features.
        """
        if USE_FLOAT64:
            X = X.astype(np.float64)

        return self.pipeline.fit_transform(X)


def preprocess_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the RobustPreprocessor pipeline to the datasets.
    Handles caching of the transformed data to disk to optimize runtime.

    The pipeline is fit ONLY on X_train (Inductive Preprocessing).

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_val (np.ndarray): Validation feature matrix.
        X_test (np.ndarray): Test feature matrix.
        load_cached_data (bool): If True, attempts to load transformed data from cache.

    Returns:
        tuple: (X_train_transformed, X_val_transformed, X_test_transformed)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # 1. Try Loading from Cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading transformed features from cache...")
        try:
            X_train_trans = np.load(cache_files["X_train"])
            X_val_trans = np.load(cache_files["X_val"])
            X_test_trans = np.load(cache_files["X_test"])
            return X_train_trans, X_val_trans, X_test_trans
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing from scratch...")

    # 2. Compute from Scratch
    print("Fitting and applying preprocessing pipeline...")

    preprocessor = RobustPreprocessor()

    # Fit on Train, Transform everything
    # We use fit_transform on train for potential efficiency, though functionally equivalent to fit().transform()
    X_train_trans = preprocessor.fit_transform(X_train)

    # Apply the learned transformation to validation and test sets
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # 3. Save to Cache
    print("Saving transformed features to cache...")
    try:
        np.save(cache_files["X_train"], X_train_trans)
        np.save(cache_files["X_val"], X_val_trans)
        np.save(cache_files["X_test"], X_test_trans)
    except Exception as e:
        print(f"Warning: Could not save preprocessing cache: {e}")

    return X_train_trans, X_val_trans, X_test_trans
