import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.feature_engine import FeatureEngine


class DataManager:
    """
    Manages data preparation for the Multi-Resolution Dual-Stream GBDT architecture.
    Orchestrates the FeatureEngine to load, merge, and cache datasets for Stream A and Stream B.
    """

    def __init__(self, debug=False):
        """
        Initialize the DataManager.

        Args:
            debug (bool): If True, limits dataset size for rapid testing.
        """
        self.debug = debug
        self.engine = FeatureEngine()

        # Ensure working directory exists (redundant with Config but safe)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def prepare_stream_datasets(self, stream_type="A", load_cached=True):
        """
        Loads and prepares training and validation datasets for a specific stream.
        Applies logic from FeatureEngine including temporal windowing and undersampling.

        Args:
            stream_type (str): "A" for Player-Player interaction, "B" for Player-Ground impact.
            load_cached (bool): If True, attempts to load from pre-computed cache.

        Returns:
            tuple: (X_train, y_train, X_val, y_val)
                X_train (pd.DataFrame): Training features.
                y_train (np.ndarray): Training labels.
                X_val (pd.DataFrame): Validation features.
                y_val (np.ndarray): Validation labels.
        """
        # Load Training Data
        X_train, y_train, _ = self.engine.get_data(
            split="train", stream=stream_type, load_cached=load_cached
        )

        # Load Validation Data
        X_val, y_val, _ = self.engine.get_data(
            split="val", stream=stream_type, load_cached=load_cached
        )

        # Apply Debugging Limits
        if self.debug:
            limit = 1000
            if len(X_train) > limit:
                X_train = X_train.iloc[:limit]
                y_train = y_train[:limit]
            if len(X_val) > limit:
                X_val = X_val.iloc[:limit]
                y_val = y_val[:limit]

        return X_train, y_train, X_val, y_val

    def get_test_data(self, stream_type="A", load_cached=True):
        """
        Loads and prepares the test dataset for inference for a specific stream.

        Args:
            stream_type (str): "A" for Player-Player, "B" for Player-Ground.
            load_cached (bool): If True, attempts to load from pre-computed cache.

        Returns:
            tuple: (X_test, ids)
                X_test (pd.DataFrame): Test features.
                ids (np.ndarray): Corresponding contact_ids.
        """
        X_test, _, ids = self.engine.get_data(
            split="test", stream=stream_type, load_cached=load_cached
        )

        # Apply Debugging Limits
        if self.debug:
            limit = 1000
            if len(X_test) > limit:
                X_test = X_test.iloc[:limit]
                ids = ids[:limit]

        return X_test, ids

    def load_merged_data(self, split="train", stream="A"):
        """
        Helper method to access the raw merged data logic if needed.
        In this architecture, this is handled internally by FeatureEngine.get_data.
        This method is provided for API completeness regarding the task description.
        """
        return self.engine.get_data(split=split, stream=stream, load_cached=False)
