import os
import pandas as pd
import numpy as np
from library.config import Config
from library.feature_manager import FeatureManager
from library.preprocessor import RobustPipeline, process_and_cache_data


class DataLoader:
    """
    Orchestrates the loading, merging, and preprocessing of data.
    Integrates FeatureManager for raw feature extraction and RobustPipeline for
    statistically robust transformation, ensuring float64 precision.
    """

    def __init__(self):
        self.feature_manager = FeatureManager()

    def load_data(self, load_cached_data=True):
        """
        Loads the training, validation, and test datasets.
        Performs feature extraction (tabular + morphological) and preprocessing
        (Yeo-Johnson + Standard Scaling).

        Args:
            load_cached_data (bool): Whether to use cached data if available.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
                - X_train, X_val, X_test: np.ndarray (float64) - Transformed features.
                - y_train, y_val: pd.Series - Target labels.
                - test_ids: pd.Series - IDs for the test set.
        """
        # 1. Get Configuration Hash to ensure cache consistency
        # We access the protected method because the data loader needs to sync
        # its transformed data cache with the feature manager's raw data cache.
        config_hash = self.feature_manager._get_config_hash()

        print(f"DataLoader: Loading data with config hash {config_hash}...")

        # 2. Load Raw Merged Datasets via FeatureManager
        # This handles the extraction of morphological features and merging with tabular data
        X_train_raw, y_train, _ = self.feature_manager.get_dataset(
            "train", load_cached_data=load_cached_data
        )
        X_val_raw, y_val, _ = self.feature_manager.get_dataset(
            "val", load_cached_data=load_cached_data
        )
        X_test_raw, _, test_ids = self.feature_manager.get_dataset(
            "test", load_cached_data=load_cached_data
        )

        # 3. Initialize Preprocessing Pipeline
        # We use a fresh pipeline instance. The fitting logic is handled by process_and_cache_data.
        pipeline = RobustPipeline()

        # 4. Process and Cache Training Data
        # We MUST fit the pipeline on the training data.
        train_cache_name = f"X_train_transformed_{config_hash}"
        X_train = process_and_cache_data(
            X_train_raw,
            pipeline,
            train_cache_name,
            fit=True,
            load_cached_data=load_cached_data,
        )

        # 5. Process and Cache Validation Data
        # Transform using the pipeline fitted on training data.
        val_cache_name = f"X_val_transformed_{config_hash}"
        X_val = process_and_cache_data(
            X_val_raw,
            pipeline,
            val_cache_name,
            fit=False,
            load_cached_data=load_cached_data,
        )

        # 6. Process and Cache Test Data
        # Transform using the pipeline fitted on training data.
        test_cache_name = f"X_test_transformed_{config_hash}"
        X_test = process_and_cache_data(
            X_test_raw,
            pipeline,
            test_cache_name,
            fit=False,
            load_cached_data=load_cached_data,
        )

        print("DataLoader: Data loading and preprocessing complete.")

        return X_train, y_train, X_val, y_val, X_test, test_ids
