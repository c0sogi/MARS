import os
import json
import pandas as pd
import numpy as np
from library.utils import FeatureLoader, INPUT_DIR, METADATA_DIR, set_seed


class DataLoader(FeatureLoader):
    """
    DataLoader module for the Pizza Request dataset.
    Extends FeatureLoader to reuse embedding generation and feature extraction logic
    while providing specific data loading and parsing functionalities.
    """

    def __init__(self, cache_dir=None):
        """
        Initialize the DataLoader.

        Args:
            cache_dir (str, optional): Directory to store cached files.
                                     Defaults to FeatureLoader's default configuration.
        """
        # Set seed for reproducibility across all operations
        set_seed(42)

        # Initialize parent FeatureLoader to setup cache directories
        if cache_dir:
            super().__init__(cache_dir=cache_dir)
        else:
            super().__init__()

    def load_data(self, debug_limit=None):
        """
        Loads the raw JSON datasets and metadata CSV files.

        Args:
            debug_limit (int, optional): If provided, limits the number of JSON entries loaded
                                       for debugging purposes.

        Returns:
            tuple: (train_json, test_json, train_meta, val_meta, test_meta)
        """
        train_path = os.path.join(INPUT_DIR, "train.json")
        test_path = os.path.join(INPUT_DIR, "test.json")

        # Load raw JSON files using parent helper
        train_json = self._load_json(train_path)
        test_json = self._load_json(test_path)

        # Apply debug limit if requested to reduce dataset size
        if debug_limit is not None:
            train_json = train_json[:debug_limit]
            test_json = test_json[:debug_limit]

        # Load Metadata CSVs defining the splits
        train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        return train_json, test_json, train_meta, val_meta, test_meta

    def extract_text_data(self, data):
        """
        Extracts and combines title and edit-aware text from the data entries.
        Delegates to FeatureLoader's internal logic to ensure consistency with
        the embedding generation process.

        Args:
            data (list): List of dictionary entries from JSON.

        Returns:
            list: List of string texts (Title + Body).
        """
        return self._get_text_data(data)

    def extract_metadata(self, data):
        """
        Extracts numerical metadata features including Unix timestamp and other
        robust signals. Delegates to FeatureLoader's internal logic.

        Args:
            data (list): List of dictionary entries from JSON.

        Returns:
            np.ndarray: Array of numerical features (float32).
        """
        return self._get_metadata_features(data)
