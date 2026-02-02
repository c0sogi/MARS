import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config
from library.utils import ensure_float64, save_to_cache, load_from_cache


class RobustPipeline:
    """
    A robust preprocessing pipeline that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling. It strictly enforces float64 precision to
    avoid numerical instability and metric floors in downstream linear discriminants.
    """

    def __init__(self):
        # Initialize transformers
        # standardize=False for PowerTransformer because we apply StandardScaler explicitly
        # This allows for modular control and inspection of each step
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.ss = StandardScaler()
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Fits the pipeline to the data.

        Args:
            X: Input data (array-like).
            y: Target values (ignored).

        Returns:
            self
        """
        # Enforce float64 precision for input
        X_float = ensure_float64(X)

        # Fit PowerTransformer (estimates lambdas for stabilizing variance)
        self.pt.fit(X_float)

        # Transform to get intermediate state for StandardScaler fitting
        # We perform this transform to compute the mean/std of the stabilized data
        X_pt = self.pt.transform(X_float)

        # Fit StandardScaler (computes mean and std)
        self.ss.fit(X_pt)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.

        Args:
            X: Input data (array-like).

        Returns:
            np.ndarray: Transformed data in float64.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RobustPipeline must be fitted before calling transform."
            )

        # Enforce float64 precision
        X_float = ensure_float64(X)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X_float)

        # Apply StandardScaler
        X_ss = self.ss.transform(X_pt)

        # Ensure final output is float64
        return ensure_float64(X_ss)

    def fit_transform(self, X, y=None):
        """
        Fits the pipeline and transforms the data in one step.

        Args:
            X: Input data.
            y: Target values.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)


def process_and_cache_data(X, pipeline, cache_name, fit=False, load_cached_data=True):
    """
    Handles the transformation of data with caching support.

    This function manages the lifecycle of transforming a dataset:
    1. It enforces float64 precision.
    2. If fit=True, it fits the provided pipeline object to X.
    3. It checks for a cached version of the transformed data.
    4. If not cached, it transforms X and saves the result to disk.

    Args:
        X (pd.DataFrame or np.ndarray): Raw input features.
        pipeline (RobustPipeline): The pipeline instance.
        cache_name (str): Unique identifier for the cache file (e.g., 'X_train_transformed').
        fit (bool): Whether to fit the pipeline on this data (True for training set).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The transformed data matrix (float64).
    """
    # Enforce input precision immediately
    X_float = ensure_float64(X)

    # If fitting is required, we must fit the pipeline object regardless of cache.
    # We cannot rely solely on cached data because the pipeline object's state
    # (lambdas, means) is needed for subsequent transforms (e.g., test set).
    if fit:
        print(f"RobustPipeline: Fitting pipeline on data for '{cache_name}'...")
        pipeline.fit(X_float)

    # Try loading transformed data from cache
    if load_cached_data:
        cached_data = load_from_cache(cache_name, expected_type="numpy")
        if cached_data is not None:
            print(f"RobustPipeline: Loaded transformed data '{cache_name}' from cache.")
            return ensure_float64(cached_data)

    # If not cached or if we just fitted and want to ensure consistency, transform now.
    print(f"RobustPipeline: Transforming data for '{cache_name}'...")
    X_transformed = pipeline.transform(X_float)

    # Save to cache for future runs
    save_to_cache(cache_name, X_transformed)
    print(f"RobustPipeline: Saved transformed data '{cache_name}' to cache.")

    return X_transformed
