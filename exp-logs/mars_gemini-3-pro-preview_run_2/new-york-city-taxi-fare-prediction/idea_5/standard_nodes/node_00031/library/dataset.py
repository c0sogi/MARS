import numpy as np
import pandas as pd
from library.config import Config
from library.data_processor import DataProcessor


class TaxiDataset:
    """
    Dataset wrapper for the NYC Taxi Fare Prediction task.
    Loads processed data and provides access to features and targets for XGBoost.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache or re-process.
        """
        self.split = split

        # Initialize DataProcessor
        processor = DataProcessor()

        # Load all data splits
        train_df, val_df, test_df = processor.process_data(
            load_cached_data=load_cached_data
        )

        # Select the appropriate dataframe
        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
            )

        # Retrieve column definitions
        self.feature_cols = processor.continuous_cols + processor.categorical_cols
        self.target_col = Config.TARGET_COL

        print(f"Dataset ({split}) loaded. Shape: {self.df.shape} samples.")

    def get_data(self):
        """
        Returns X (features) and y (target) as numpy arrays.
        For test set, y is None.
        Also returns keys for test set.
        """
        X = self.df[self.feature_cols].values.astype(np.float32)

        if self.split != "test":
            y = self.df[self.target_col].values.astype(np.float32)
            return X, y
        else:
            keys = self.df["key"].astype(str).values
            return X, None, keys
