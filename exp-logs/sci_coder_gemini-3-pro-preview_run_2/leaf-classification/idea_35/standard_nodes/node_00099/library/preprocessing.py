import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library.config import (
    CACHE_DIR,
    FLOAT_PRECISION,
    RANDOM_SEED,
    POWER_TRANSFORM_METHOD,
    QUANTILE_TRANSFORM_N_QUANTILES,
    QUANTILE_TRANSFORM_OUTPUT_DIST,
)
from library.data_loader import get_combined_dataset


class StereoscopicPreprocessor:
    """
    Implements the Stereoscopic Preprocessing Architecture.
    Maintains three parallel pipelines:
    1. Parametric Global: Yeo-Johnson on 192 global features.
    2. Regularized Rank Global: QuantileTransformer (n=30) on 192 global features.
    3. Orthogonal Morphometric: Yeo-Johnson on 11 macro features.
    """

    def __init__(self):
        # Pipeline A: Parametric Gaussian Anchors
        self.pt_global = PowerTransformer(method=POWER_TRANSFORM_METHOD)

        # Pipeline B: Regularized Non-Parametric Experts
        # Strictly constrained quantiles to prevent overfitting
        self.qt_global = QuantileTransformer(
            output_distribution=QUANTILE_TRANSFORM_OUTPUT_DIST,
            n_quantiles=QUANTILE_TRANSFORM_N_QUANTILES,
            random_state=RANDOM_SEED,
        )

        # Pipeline C: Orthogonal Morphometric Experts
        self.pt_macro = PowerTransformer(method=POWER_TRANSFORM_METHOD)

        self.is_fitted = False

    def fit(self, X):
        """
        Fits the transformers on the provided data.
        X is expected to be the combined dataset (Global + Macro).

        Args:
            X (np.ndarray): Shape (N, 203).
                            Cols 0-191: Global Features.
                            Cols 192-202: Macro Features.
        """
        # Split features
        X_global = X[:, :192]
        X_macro = X[:, 192:]

        # Fit pipelines
        self.pt_global.fit(X_global)
        self.qt_global.fit(X_global)
        self.pt_macro.fit(X_macro)

        self.is_fitted = True
        return self

    def transform(self, X, view):
        """
        Transforms the data according to the requested view.

        Args:
            X (np.ndarray): Shape (N, 203). Combined dataset.
            view (str): One of 'global_parametric', 'global_rank', 'macro'.

        Returns:
            np.ndarray: Transformed feature matrix in float64.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        # Ensure input is float64 before processing to minimize numerical noise
        X = X.astype(FLOAT_PRECISION)

        X_global = X[:, :192]
        X_macro = X[:, 192:]

        if view == "global_parametric":
            # Pipeline A
            X_trans = self.pt_global.transform(X_global)
        elif view == "global_rank":
            # Pipeline B
            X_trans = self.qt_global.transform(X_global)
        elif view == "macro":
            # Pipeline C
            X_trans = self.pt_macro.transform(X_macro)
        else:
            raise ValueError(f"Unknown view: {view}")

        return X_trans.astype(FLOAT_PRECISION)


def get_transformed_data(split, view, load_cached_data=True):
    """
    Retrieves the transformed dataset for a specific split and view.
    Handles caching and ensures transformers are strictly fitted on the training set.

    Args:
        split (str): 'train', 'val', or 'test'.
        view (str): 'global_parametric', 'global_rank', or 'macro'.
        load_cached_data (bool): Whether to use disk cache.

    Returns:
        tuple: (X_transformed, y, ids)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(CACHE_DIR, f"X_{split}_{view}.npy")

    # Load y and ids from the data loader (these don't change with transformation)
    # We use the data loader to get the raw data anyway if cache misses
    _, y, ids = get_combined_dataset(split, load_cached_data=True)

    # 1. Try to load X from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached transformed data for split='{split}', view='{view}'...")
        X_transformed = np.load(cache_path)
        return X_transformed, y, ids

    # 2. Compute if cache miss
    print(f"Computing transformed data for split='{split}', view='{view}'...")

    # Always load raw training data to fit the transformers (Leakage Prevention)
    X_train_raw, _, _ = get_combined_dataset("train", load_cached_data=True)

    # Initialize and fit preprocessor
    preprocessor = StereoscopicPreprocessor()
    preprocessor.fit(X_train_raw)

    # Load raw target data
    # If split is train, we already have it. If not, load it.
    if split == "train":
        X_target_raw = X_train_raw
    else:
        X_target_raw, _, _ = get_combined_dataset(split, load_cached_data=True)

    # Transform
    X_transformed = preprocessor.transform(X_target_raw, view)

    # 3. Save to cache
    print(f"Saving transformed data to {cache_path}")
    np.save(cache_path, X_transformed)

    return X_transformed, y, ids
