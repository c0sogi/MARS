import numpy as np
import pandas as pd
from library.feature_extraction import load_data
from library.config import FLOAT_PRECISION, DEBUG_SAMPLE_SIZE


class LeafDataLoader:
    """
    Data loader class responsible for the ingestion and fusion of tabular and image data.
    It leverages the feature extraction library to load data, enforces high-precision
    numerical formats, and handles debug sampling.
    """

    def __init__(self, debug_sample_size=DEBUG_SAMPLE_SIZE):
        """
        Initialize the LeafDataLoader.

        Args:
            debug_sample_size (int, optional): The number of samples to load.
                                               If None, loads the full dataset.
                                               Defaults to DEBUG_SAMPLE_SIZE from config.
        """
        self.debug_sample_size = debug_sample_size

    def get_train_data(self, load_cached_data=True):
        """
        Loads the training dataset.

        Args:
            load_cached_data (bool): If True, attempts to load features from cache.

        Returns:
            tuple: (X, y, ids)
                X (np.ndarray): Feature matrix of shape (n_samples, n_features) in float64.
                y (np.ndarray): Target labels of shape (n_samples,).
                ids (np.ndarray): Image IDs of shape (n_samples,).
        """
        return self._load_split("train", load_cached_data)

    def get_val_data(self, load_cached_data=True):
        """
        Loads the validation dataset.

        Args:
            load_cached_data (bool): If True, attempts to load features from cache.

        Returns:
            tuple: (X, y, ids)
                X (np.ndarray): Feature matrix of shape (n_samples, n_features) in float64.
                y (np.ndarray): Target labels of shape (n_samples,).
                ids (np.ndarray): Image IDs of shape (n_samples,).
        """
        return self._load_split("val", load_cached_data)

    def get_test_data(self, load_cached_data=True):
        """
        Loads the test dataset.

        Args:
            load_cached_data (bool): If True, attempts to load features from cache.

        Returns:
            tuple: (X, ids)
                X (np.ndarray): Feature matrix of shape (n_samples, n_features) in float64.
                ids (np.ndarray): Image IDs of shape (n_samples,).
        """
        X, _, ids = self._load_split("test", load_cached_data)
        return X, ids

    def _load_split(self, split, load_cached_data):
        """
        Internal helper to load, format, and sample data for a specific split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use the caching mechanism.

        Returns:
            tuple: (X, y, ids)
        """
        # 1. Load data using the library function.
        # This handles reading metadata, extracting geometric features from images,
        # combining them with tabular features, and managing the Parquet/Numpy cache.
        X_df, y, ids = load_data(split=split, load_cached_data=load_cached_data)

        # 2. Convert to high-precision NumPy arrays.
        # The DataFrame columns are already sorted (tabular alphanumerically + geometric fixed order)
        # ensuring a deterministic memory layout.
        X = X_df.values.astype(FLOAT_PRECISION)

        # 3. Apply debug sampling if configured.
        if self.debug_sample_size is not None and self.debug_sample_size > 0:
            if len(X) > self.debug_sample_size:
                X = X[: self.debug_sample_size]
                ids = ids[: self.debug_sample_size]
                if y is not None:
                    y = y[: self.debug_sample_size]

        return X, y, ids
