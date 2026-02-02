import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.utils import seed_everything


class DataManager:
    """
    Manages data loading, splitting, and sampling for the dual-stream model architecture.
    Delegates complex feature engineering and caching to the FeatureEngineer class.
    """

    def __init__(self, config=Config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)

    def _undersample(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Applies random undersampling to the majority class (0) to achieve
        the ratio defined in Config.UNDERSAMPLE_RATIO.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Undersampled X and y.
        """
        # Ensure reproducibility
        np.random.seed(self.config.SEED)

        indices = np.arange(len(y))
        pos_indices = indices[y == 1]
        neg_indices = indices[y == 0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate target number of negatives based on ratio
        target_neg = int(n_pos * self.config.UNDERSAMPLE_RATIO)

        if target_neg < n_neg:
            sampled_neg_indices = np.random.choice(
                neg_indices, size=target_neg, replace=False
            )
        else:
            sampled_neg_indices = neg_indices

        # Combine positive and sampled negative indices
        keep_indices = np.concatenate([pos_indices, sampled_neg_indices])

        # Shuffle the resulting dataset
        np.random.shuffle(keep_indices)

        # Subset data
        X_sampled = X.iloc[keep_indices].reset_index(drop=True)
        y_sampled = y[keep_indices]

        return X_sampled, y_sampled

    def get_train_data(
        self, load_cached_data: bool = True
    ) -> Dict[str, Dict[str, Tuple[pd.DataFrame, np.ndarray]]]:
        """
        Retrieves training and validation data for both Stream A and Stream B.
        Applies random undersampling to the training sets to handle class imbalance.

        Args:
            load_cached_data (bool): Whether to load features from parquet cache if available.

        Returns:
            Dict: A nested dictionary containing train/val sets for both streams.
                Structure:
                {
                    "stream_A": {"train": (X, y), "val": (X, y)},
                    "stream_B": {"train": (X, y), "val": (X, y)}
                }
        """
        # Generate or load features using the FeatureEngineer
        # This handles the robust fusion logic and caching
        (train_A, val_A, train_B, val_B) = (
            self.feature_engineer.generate_train_features(
                load_cached_data=load_cached_data
            )
        )

        X_train_A, y_train_A = train_A
        X_val_A, y_val_A = val_A
        X_train_B, y_train_B = train_B
        X_val_B, y_val_B = val_B

        # Apply Undersampling to Training Data
        # Validation data is left untouched to provide accurate performance metrics
        print(f"Undersampling Stream A (Original size: {len(y_train_A)})...")
        X_train_A_samp, y_train_A_samp = self._undersample(X_train_A, y_train_A)
        print(
            f"Stream A Train size after sampling: {len(y_train_A_samp)} (Pos: {sum(y_train_A_samp)})"
        )

        print(f"Undersampling Stream B (Original size: {len(y_train_B)})...")
        X_train_B_samp, y_train_B_samp = self._undersample(X_train_B, y_train_B)
        print(
            f"Stream B Train size after sampling: {len(y_train_B_samp)} (Pos: {sum(y_train_B_samp)})"
        )

        return {
            "stream_A": {
                "train": (X_train_A_samp, y_train_A_samp),
                "val": (X_val_A, y_val_A),
            },
            "stream_B": {
                "train": (X_train_B_samp, y_train_B_samp),
                "val": (X_val_B, y_val_B),
            },
        }

    def get_test_data(
        self, load_cached_data: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Retrieves test data for both streams.

        Args:
            load_cached_data (bool): Whether to load features from parquet cache if available.

        Returns:
            Tuple: (X_test_A, ids_A, X_test_B, ids_B)
        """
        return self.feature_engineer.generate_test_features(
            load_cached_data=load_cached_data
        )
