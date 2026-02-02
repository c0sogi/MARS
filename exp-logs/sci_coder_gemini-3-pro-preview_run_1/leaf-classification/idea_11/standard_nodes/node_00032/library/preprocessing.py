import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config
from library import data_loader


class GaussianPipeline:
    """
    A preprocessing pipeline that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling to Gaussianize features for LDA.
    """

    def __init__(self):
        # standardize=False is critical because we apply StandardScaler explicitly afterwards.
        # This avoids double-standardization and allows for specific control over the
        # normalization statistics.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the pipeline to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.
            y (np.ndarray, optional): Target labels (ignored).

        Returns:
            self
        """
        self.pt.fit(X)
        # Transform to feed into scaler
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Applies the transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Transformed features.
        """
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.
            y (np.ndarray, optional): Target labels (ignored).

        Returns:
            np.ndarray: Transformed features.
        """
        self.fit(X, y)
        return self.transform(X)


def get_fitted_pipeline(load_cached_data=True):
    """
    Creates and fits the GaussianPipeline on the full training dataset.

    Args:
        load_cached_data (bool): Whether to use cached raw training data via data_loader.

    Returns:
        GaussianPipeline: The fitted pipeline object.
    """
    # Load raw training data
    # We use the data_loader to ensure schema consistency
    X_train, _, _ = data_loader.load_dataset("train", load_cached_data=load_cached_data)

    # Initialize and fit pipeline
    pipeline = GaussianPipeline()
    pipeline.fit(X_train)

    return pipeline


def get_transformed_data(split, pipeline=None, load_cached_data=True):
    """
    Retrieves the transformed dataset for a given split.
    Uses caching to store the processed numpy arrays to disk, avoiding re-computation.

    Args:
        split (str): 'train', 'val', or 'test'.
        pipeline (GaussianPipeline, optional): The fitted pipeline to use for transformation.
                                               Required if data is not found in cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_transformed, y, ids)
               X_transformed (np.ndarray): The processed feature matrix.
               y (np.ndarray or None): The target labels.
               ids (np.ndarray): The image identifiers.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache path for the transformed feature matrix
    cache_path = os.path.join(config.WORKING_DIR, f"X_{split}_transformed.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # Load transformed features
        X_transformed = np.load(cache_path)

        # Load y and ids using data_loader (which handles its own caching efficiently)
        # We don't need the raw X here, so we ignore it
        _, y, ids = data_loader.load_dataset(split, load_cached_data=load_cached_data)

        return X_transformed, y, ids

    # If not in cache or reload forced, compute from scratch
    if pipeline is None:
        # We cannot transform without a pipeline.
        # Note: Even for 'train', we require the caller to provide the pipeline
        # (via get_fitted_pipeline) to ensure the object state is managed explicitly.
        raise ValueError(
            f"Pipeline object is required to transform '{split}' data when cache is missing."
        )

    # Load raw data
    X_raw, y, ids = data_loader.load_dataset(split, load_cached_data=load_cached_data)

    # Transform
    X_transformed = pipeline.transform(X_raw)

    # Save to cache
    np.save(cache_path, X_transformed)

    return X_transformed, y, ids
