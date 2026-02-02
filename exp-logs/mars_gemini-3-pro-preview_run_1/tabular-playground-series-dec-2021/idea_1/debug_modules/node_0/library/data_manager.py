import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Add library path to allow imports from utils.py
sys.path.append("./library")
from utils import load_data, set_seed


class DataManager:
    """
    Manages data loading, preprocessing, and formatting for the XGBoost pipeline.
    Handles caching via the utility library and ensures correct data formats (DMatrix).
    """

    def __init__(self, data_dir="./metadata", cache_dir="./working/idea_1"):
        """
        Initialize the DataManager.

        Args:
            data_dir (str): Directory containing the metadata CSV files.
            cache_dir (str): Directory to store/retrieve cached Parquet files.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.le = LabelEncoder()
        self.target_col = "Cover_Type"
        self.id_col = "Id"
        self.feature_cols = []

    def load_dataset(self, load_cached_data=True, sample_size=None):
        """
        Loads the train, validation, and test datasets.
        Leverages the library.utils.load_data function for caching and sampling.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            sample_size (int, optional): If provided, returns a random sample for debugging.
        """
        # Load datasets using the provided utility function
        self.train_df = load_data(
            "train", load_cached_data, self.data_dir, self.cache_dir, sample_size
        )
        self.val_df = load_data(
            "val", load_cached_data, self.data_dir, self.cache_dir, sample_size
        )
        self.test_df = load_data(
            "test", load_cached_data, self.data_dir, self.cache_dir, sample_size
        )

        # Identify feature columns: All columns except Target and Id
        if self.train_df is not None:
            all_cols = self.train_df.columns.tolist()
            self.feature_cols = [
                c for c in all_cols if c != self.target_col and c != self.id_col
            ]

    def encode_target(self):
        """
        Encodes the target variable 'Cover_Type' into 0-indexed integers.
        Fits the encoder on the training set and transforms both training and validation sets.
        """
        if self.train_df is None:
            raise ValueError("Training data not loaded. Call load_dataset() first.")

        # Fit LabelEncoder on training targets
        y_train = self.train_df[self.target_col].values
        self.le.fit(y_train)

        # Transform training targets
        self.train_df[self.target_col] = self.le.transform(y_train)

        # Transform validation targets if available
        if self.val_df is not None and self.target_col in self.val_df.columns:
            y_val = self.val_df[self.target_col].values
            self.val_df[self.target_col] = self.le.transform(y_val)

    def get_dmatrix(self, split="train"):
        """
        Converts the specified data split into an XGBoost DMatrix.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            xgb.DMatrix: The DMatrix object ready for XGBoost.
        """
        if split == "train":
            if self.train_df is None:
                raise ValueError("Train data not loaded.")
            X = self.train_df[self.feature_cols]
            y = self.train_df[self.target_col]
            return xgb.DMatrix(X, label=y)

        elif split == "val":
            if self.val_df is None:
                raise ValueError("Validation data not loaded.")
            X = self.val_df[self.feature_cols]
            y = self.val_df[self.target_col]
            return xgb.DMatrix(X, label=y)

        elif split == "test":
            if self.test_df is None:
                raise ValueError("Test data not loaded.")
            X = self.test_df[self.feature_cols]
            # Test set has no labels
            return xgb.DMatrix(X)

        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

    def inverse_transform_target(self, predictions):
        """
        Converts model predictions (integers) back to original class labels.

        Args:
            predictions (array-like): Predicted class indices.

        Returns:
            np.array: Original class labels.
        """
        return self.le.inverse_transform(predictions.astype(int))

    def get_test_ids(self):
        """
        Returns the 'Id' column from the test set for submission formatting.

        Returns:
            np.array: Array of test IDs.
        """
        if self.test_df is None:
            raise ValueError("Test data not loaded.")
        return self.test_df[self.id_col].values
