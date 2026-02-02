import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
    TARGET_COL,
)

# Note: TARGET_COL is not in config.py provided in prompt, but 'fare_amount' is standard.
# I will use string literal "fare_amount" as per dataset description.
from library.feature_engineering import process_dataset


class TaxiDataLoader:
    """
    Data Loader class that orchestrates the loading, cleaning, and feature engineering
    of the NYC Taxi dataset.
    """

    def __init__(self, debug_mode=DEBUG_MODE, load_cached_data=True):
        self.debug_mode = debug_mode
        self.load_cached_data = load_cached_data
        self.target_col = "fare_amount"
        self.key_col = "key"

    def get_train_data(self):
        """
        Loads and processes the training data.
        Applies strict filtering (bounding box + fare range).
        Returns:
            X_train (pd.DataFrame): Feature matrix.
            y_train (pd.Series): Target variable.
        """
        df = process_dataset(
            input_path=TRAIN_DATA_PATH,
            is_train=True,
            load_cached_data=self.load_cached_data,
            debug_mode=self.debug_mode,
        )

        # Separate Features and Target
        if self.target_col in df.columns:
            y_train = df[self.target_col]
            X_train = df.drop(columns=[self.target_col, self.key_col], errors="ignore")
        else:
            raise ValueError(
                f"Target column '{self.target_col}' not found in training data."
            )

        return X_train, y_train

    def get_val_data(self):
        """
        Loads and processes the validation data.
        Applies strict filtering (bounding box) to ensure clean evaluation metrics.
        Returns:
            X_val (pd.DataFrame): Feature matrix.
            y_val (pd.Series): Target variable.
        """
        # We treat validation data as 'train' for the purpose of process_dataset
        # to ensure the bounding box filters are applied, removing outliers
        # that would skew the validation RMSE.
        df = process_dataset(
            input_path=VAL_DATA_PATH,
            is_train=True,
            load_cached_data=self.load_cached_data,
            debug_mode=self.debug_mode,
        )

        if self.target_col in df.columns:
            y_val = df[self.target_col]
            X_val = df.drop(columns=[self.target_col, self.key_col], errors="ignore")
        else:
            raise ValueError(
                f"Target column '{self.target_col}' not found in validation data."
            )

        return X_val, y_val

    def get_test_data(self):
        """
        Loads and processes the test data.
        Does NOT drop rows (is_train=False) to ensure alignment with submission file.
        Returns:
            X_test (pd.DataFrame): Feature matrix.
            keys (pd.Series): The 'key' column required for submission.
        """
        df = process_dataset(
            input_path=TEST_DATA_PATH,
            is_train=False,
            load_cached_data=self.load_cached_data,
            debug_mode=self.debug_mode,
        )

        # Extract keys for submission
        if self.key_col in df.columns:
            keys = df[self.key_col]
        else:
            raise ValueError(f"Key column '{self.key_col}' not found in test data.")

        # Drop key column from features
        X_test = df.drop(columns=[self.key_col], errors="ignore")

        # Ensure target is not in X (it shouldn't be in test, but just in case)
        X_test = X_test.drop(columns=[self.target_col], errors="ignore")

        return X_test, keys
