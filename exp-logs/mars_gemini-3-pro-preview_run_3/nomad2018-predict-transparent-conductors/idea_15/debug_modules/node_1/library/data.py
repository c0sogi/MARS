import numpy as np
import pandas as pd
import os
from library.config import Config
from library.features import process_dataset


class TargetTransformer:
    """
    Handles logarithmic transformation of target variables to stabilize variance
    and improve model performance for skewed targets.
    Transformation: z = log(1 + y)
    Inverse: y = exp(z) - 1
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Apply log(1 + y) transformation.
        Args:
            y (array-like): Original target values.
        Returns:
            array-like: Log-transformed values.
        """
        return np.log1p(y)

    def inverse_transform(self, z):
        """
        Apply exp(z) - 1 transformation.
        Args:
            z (array-like): Log-transformed predictions.
        Returns:
            array-like: Original scale predictions.
        """
        return np.expm1(z)


def load_and_featurize_data(debug_sample=None, load_cached_data=True):
    """
    Orchestrates the loading and featurization of Train, Validation, and Test datasets.
    Utilizes the caching mechanism implemented in library.features.process_dataset.

    Args:
        debug_sample (int, optional): If set, limits the number of rows processed for debugging.
        load_cached_data (bool): If True, attempts to load features from Parquet cache.
                                 If False or cache missing, recomputes features.

    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames containing metadata and computed features.
    """

    # Ensure working directory exists (redundant with Config but safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Process Training Data
    print(f"--- Loading/Processing Training Data (Sample={debug_sample}) ---")
    train_df = process_dataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_file=Config.TRAIN_FEATS_FILE,
        load_cached_data=load_cached_data,
        debug_sample=debug_sample,
    )

    # Process Validation Data
    print(f"--- Loading/Processing Validation Data (Sample={debug_sample}) ---")
    val_df = process_dataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_file=Config.VAL_FEATS_FILE,
        load_cached_data=load_cached_data,
        debug_sample=debug_sample,
    )

    # Process Test Data
    print(f"--- Loading/Processing Test Data (Sample={debug_sample}) ---")
    test_df = process_dataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_file=Config.TEST_FEATS_FILE,
        load_cached_data=load_cached_data,
        debug_sample=debug_sample,
    )

    return train_df, val_df, test_df
