import os
import pandas as pd
from library.config import Config


class DataManager:
    """
    Handles data loading and schema alignment to prevent leakage for the Pizza Request dataset.
    """

    def __init__(self):
        """
        Initialize the DataManager with configuration settings.
        """
        self.config = Config()

    def load_data(self, debug_size=None):
        """
        Loads the training, validation, and test datasets from the metadata CSV files.

        Args:
            debug_size (int, optional): If provided, limits the number of rows loaded
                                        for debugging purposes. Defaults to None.

        Returns:
            tuple: A tuple containing (train_df, val_df, test_df).
        """
        # Define paths
        train_path = self.config.TRAIN_DATA_PATH
        val_path = self.config.VAL_DATA_PATH
        test_path = self.config.TEST_DATA_PATH

        # Validate paths
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data file not found: {val_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found: {test_path}")

        # Load datasets
        # Using pandas read_csv. We load the full dataset first to ensure consistent types,
        # then slice if debug_size is set.
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Apply debug sampling if requested
        if debug_size is not None and debug_size > 0:
            train_df = train_df.head(debug_size)
            val_df = val_df.head(debug_size)
            test_df = test_df.head(debug_size)

        return train_df, val_df, test_df

    def align_datasets(self, train_df, val_df, test_df):
        """
        Aligns the schemas of the training and validation datasets with the test dataset.
        This ensures that only features available at inference time (present in test)
        are used for training, effectively preventing data leakage from '_at_retrieval' columns.

        Args:
            train_df (pd.DataFrame): The training dataframe.
            val_df (pd.DataFrame): The validation dataframe.
            test_df (pd.DataFrame): The test dataframe.

        Returns:
            tuple: A tuple containing the aligned (train_df, val_df, test_df).
        """
        # 1. Identify valid features based on Test set (Ground Truth for availability)
        test_features = set(test_df.columns)

        # 2. Identify available features in Train set
        train_features = set(train_df.columns)

        # 3. Find intersection (Features present in both)
        # This automatically drops columns like 'number_of_upvotes_of_request_at_retrieval'
        # which are in Train but not in Test.
        common_features = test_features.intersection(train_features)

        # 4. Handle Target Variable
        # The target variable is likely not in Test, but must be kept in Train/Val.
        target_col = "requester_received_pizza"

        # Create the list of columns to keep for training (Common + Target)
        train_cols_to_keep = list(common_features)
        if target_col in train_df.columns and target_col not in train_cols_to_keep:
            train_cols_to_keep.append(target_col)

        # Create the list of columns to keep for testing (Common only)
        test_cols_to_keep = list(common_features)

        # Sort for consistency
        train_cols_to_keep.sort()
        test_cols_to_keep.sort()

        # 5. Filter DataFrames
        aligned_train = train_df[train_cols_to_keep].copy()
        aligned_val = val_df[train_cols_to_keep].copy()
        aligned_test = test_df[test_cols_to_keep].copy()

        return aligned_train, aligned_val, aligned_test

    def get_data(self, debug_size=None):
        """
        High-level method to load and align all datasets.

        Args:
            debug_size (int, optional): Limit dataset size for debugging.

        Returns:
            tuple: (aligned_train_df, aligned_val_df, aligned_test_df)
        """
        # Load raw data
        train_df, val_df, test_df = self.load_data(debug_size=debug_size)

        # Align schemas to prevent leakage
        train_df, val_df, test_df = self.align_datasets(train_df, val_df, test_df)

        return train_df, val_df, test_df
