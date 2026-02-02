import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler

from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    FLOAT_PRECISION,
)
from library.utils import get_logger
from library.features import generate_feature_set

logger = get_logger("data")


class SanitizedPreprocessor:
    """
    A strict preprocessing pipeline designed for high-precision linear discriminants.

    Pipeline Steps:
    1. Cast to float64 (Precision Enforcement).
    2. VarianceThreshold(0.0): Removes constant features (Sanitization Barrier).
    3. PowerTransformer(yeo-johnson): Gaussianizes features (Stabilization).
    4. StandardScaler: Centers and scales features (Normalization).

    This class ensures that the pipeline is fitted ONLY on the training data
    and applied consistently to validation and test data.
    """

    def __init__(self):
        self.variance_selector = VarianceThreshold(threshold=0.0)
        self.power_transformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._feature_names_in = None
        self._feature_names_out = None

    def fit(self, X, y=None):
        """
        Fits the preprocessing pipeline on the training data.

        Args:
            X (pd.DataFrame or np.ndarray): The training feature matrix.
            y (np.ndarray, optional): Ignored.

        Returns:
            self
        """
        # Ensure float64 precision
        X_clean = np.array(X, dtype=np.float64)

        # 1. Fit Variance Threshold (Sanitization)
        # This identifies constant columns that would cause issues downstream
        self.variance_selector.fit(X_clean)
        X_varied = self.variance_selector.transform(X_clean)

        # 2. Fit Power Transformer (Stabilization)
        # Yeo-Johnson handles positive and negative values
        self.power_transformer.fit(X_varied)
        X_power = self.power_transformer.transform(X_varied)

        # 3. Fit Standard Scaler (Normalization)
        self.scaler.fit(X_power)

        self.is_fitted = True

        # Track feature names if input is DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            # We can't easily track out names after variance threshold without extra logic,
            # but we track what we can.

        logger.info(
            f"Preprocessor fitted. "
            f"Input features: {X_clean.shape[1]}, "
            f"Kept after sanitization: {X_varied.shape[1]}"
        )
        return self

    def transform(self, X):
        """
        Applies the fitted transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): The feature matrix to transform.

        Returns:
            np.ndarray: The transformed feature matrix (float64).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "SanitizedPreprocessor must be fitted before calling transform."
            )

        # Ensure float64 precision
        X_clean = np.array(X, dtype=np.float64)

        # 1. Apply Variance Threshold
        X_varied = self.variance_selector.transform(X_clean)

        # 2. Apply Power Transformer
        X_power = self.power_transformer.transform(X_varied)

        # 3. Apply Standard Scaler
        X_scaled = self.scaler.transform(X_power)

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in one step.
        """
        return self.fit(X, y).transform(X)


def load_dataset(
    dataset_name: str, limit_size: int = None, load_cached_data: bool = True
):
    """
    Loads the requested dataset (train, val, or test), generating features if necessary.

    Args:
        dataset_name (str): One of 'train', 'val', 'test'.
        limit_size (int, optional): If provided, limits the number of rows (for debugging).
        load_cached_data (bool): Whether to attempt loading from the cache.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Raw feature matrix (Tabular + Geometric).
            y (np.ndarray or None): Target labels.
            ids (np.ndarray): Image identifiers.
    """
    # Map dataset name to metadata path
    if dataset_name == "train":
        metadata_path = TRAIN_DATA_PATH
    elif dataset_name == "val":
        metadata_path = VAL_DATA_PATH
    elif dataset_name == "test":
        metadata_path = TEST_DATA_PATH
    else:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Must be 'train', 'val', or 'test'."
        )

    logger.info(f"Requesting dataset: {dataset_name} (limit_size={limit_size})")

    # Delegate to library.features to handle generation and caching of raw features
    X, y, ids = generate_feature_set(
        metadata_path=metadata_path,
        dataset_name=dataset_name,
        load_cached_data=load_cached_data,
    )

    # Apply debugging limit if requested
    if limit_size is not None and limit_size > 0:
        logger.info(
            f"Limiting {dataset_name} dataset to {limit_size} samples for debugging."
        )
        X = X.iloc[:limit_size]
        ids = ids[:limit_size]
        if y is not None:
            y = y[:limit_size]

    return X, y, ids
