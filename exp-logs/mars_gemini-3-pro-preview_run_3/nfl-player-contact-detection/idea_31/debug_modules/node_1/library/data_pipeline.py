import pandas as pd
import numpy as np
import os
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import setup_logger


class DataPipeline:
    """
    Orchestrates the data flow for the contact detection system.
    Interfaces with FeatureEngineer to generate/load features and handles
    dataset-specific operations like undersampling for training.
    """

    def __init__(self):
        self.logger = setup_logger("DataPipeline")
        self.feature_engineer = FeatureEngineer()

    def _undersample(self, X, y, ids, ratio):
        """
        Applies targeted majority undersampling.
        Retains all positives and samples negatives to achieve the desired ratio.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target vector.
            ids (np.array): Contact IDs.
            ratio (float): Ratio of negatives to positives.

        Returns:
            tuple: (X_resampled, y_resampled, ids_resampled)
        """
        self.logger.info(f"Applying undersampling with Neg/Pos ratio: {ratio}")

        # Ensure reproducibility
        np.random.seed(Config.SEED)

        # Identify indices
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        n_pos = len(pos_indices)
        n_neg_total = len(neg_indices)

        if n_pos == 0:
            self.logger.warning("No positive samples found. Skipping undersampling.")
            return X, y, ids

        # Calculate required negatives
        n_neg_keep = int(n_pos * ratio)

        if n_neg_keep >= n_neg_total:
            self.logger.info("Undersampling skipped: Not enough negatives to reduce.")
            return X, y, ids

        # Randomly sample negatives
        neg_indices_keep = np.random.choice(neg_indices, size=n_neg_keep, replace=False)

        # Combine and shuffle
        keep_indices = np.concatenate([pos_indices, neg_indices_keep])
        np.random.shuffle(keep_indices)

        self.logger.info(
            f"Original size: {len(y)}. Undersampled size: {len(keep_indices)}. Positives: {n_pos}"
        )

        # Slice
        # X is a DataFrame, y and ids are numpy arrays
        X_res = X.iloc[keep_indices].reset_index(drop=True)
        y_res = y[keep_indices]
        ids_res = ids[keep_indices]

        return X_res, y_res, ids_res

    def load_data(self, mode, stream, load_cached_data=True):
        """
        Loads data for a specific mode (train/validation/test) and stream (streamA/streamB).
        Delegates feature engineering to the FeatureEngineer class.
        Applies undersampling if mode is 'train'.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            stream (str): 'streamA' or 'streamB'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X, y, ids)
        """
        self.logger.info(f"Requesting data for Mode: {mode}, Stream: {stream}")

        # Delegate to FeatureEngineer
        # The FeatureEngineer handles the caching of the heavy feature generation steps.
        # It strictly follows the logic: check cache -> load if exists -> compute if not -> save.
        if stream == "streamA":
            X, y, ids = self.feature_engineer.process_stream_a(
                mode=mode, load_cached_data=load_cached_data
            )
        elif stream == "streamB":
            X, y, ids = self.feature_engineer.process_stream_b(
                mode=mode, load_cached_data=load_cached_data
            )
        else:
            raise ValueError(f"Unknown stream: {stream}")

        # Apply Undersampling only for training
        if mode == "train":
            X, y, ids = self._undersample(X, y, ids, Config.NEG_POS_RATIO)

        return X, y, ids
