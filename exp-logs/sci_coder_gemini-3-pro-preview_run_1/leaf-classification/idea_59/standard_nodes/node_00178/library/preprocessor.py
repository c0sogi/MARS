import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from library import config


class SanitizedTransformer:
    """
    Implements the 'Sanitized Pipeline' for robust feature engineering.

    Steps:
    1. Sanitization Barrier: VarianceThreshold(threshold=0) to remove constant features.
    2. Stabilization: PowerTransformer(method='yeo-johnson', standardize=False).
    3. Scaling: StandardScaler.

    Enforces float64 precision throughout.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("selector", VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)),
                (
                    "stabilizer",
                    PowerTransformer(method="yeo-johnson", standardize=False),
                ),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, X):
        """
        Fits the pipeline to the data.
        Args:
            X (pd.DataFrame or np.ndarray): Training data.
        """
        # Ensure float64
        X = np.array(X, dtype=config.FLOAT_PRECISION)
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.
        Args:
            X (pd.DataFrame or np.ndarray): Data to transform.
        Returns:
            np.ndarray: Transformed data in float64.
        """
        # Ensure float64
        X = np.array(X, dtype=config.FLOAT_PRECISION)
        X_trans = self.pipeline.transform(X)
        return X_trans.astype(config.FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Fits and transforms the data.
        """
        # Ensure float64
        X = np.array(X, dtype=config.FLOAT_PRECISION)
        X_trans = self.pipeline.fit_transform(X)
        return X_trans.astype(config.FLOAT_PRECISION)


def get_transformed_data(
    X_train, X_val, X_test, debug_suffix="", load_cached_data=True
):
    """
    Orchestrates the preprocessing pipeline:
    1. Checks for cached transformed data.
    2. If not found, fits the SanitizedTransformer on X_train.
    3. Transforms X_train, X_val, and X_test.
    4. Caches the results.

    Args:
        X_train, X_val, X_test: Raw feature matrices (DataFrame or ndarray).
        debug_suffix (str): Suffix to distinguish cache files (e.g., "_debug_100").
        load_cached_data (bool): Whether to use existing cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_train = os.path.join(
        config.CACHE_DIR, f"X_train_transformed{debug_suffix}.npy"
    )
    cache_val = os.path.join(config.CACHE_DIR, f"X_val_transformed{debug_suffix}.npy")
    cache_test = os.path.join(config.CACHE_DIR, f"X_test_transformed{debug_suffix}.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            print(f"Loading transformed data from cache ({config.CACHE_DIR})...")
            try:
                X_train_trans = np.load(cache_train)
                X_val_trans = np.load(cache_val)
                X_test_trans = np.load(cache_test)
                return X_train_trans, X_val_trans, X_test_trans
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cached transformed data not found. Computing...")

    # 2. Compute from scratch
    print("Fitting SanitizedTransformer on Training Data...")
    transformer = SanitizedTransformer()

    # Fit on Train, Transform Train
    # Note: We use fit_transform on train for efficiency, though fit then transform is equivalent
    X_train_trans = transformer.fit_transform(X_train)

    # Transform Val and Test using the train-fitted pipeline (Inductive Preprocessing)
    print("Transforming Validation and Test Data...")
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # 3. Save to cache
    print(f"Caching transformed data to {config.CACHE_DIR}...")
    np.save(cache_train, X_train_trans)
    np.save(cache_val, X_val_trans)
    np.save(cache_test, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
