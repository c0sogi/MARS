import pandas as pd
import numpy as np
import warnings
from library.features import generate_features
from library.config import TARGET_COLS
from library.utils import log_transform

# Suppress warnings
warnings.filterwarnings("ignore")


class FeatureLoader:
    """
    Handles loading, preprocessing, and splitting of the dataset.
    Integrates feature extraction, target transformation, and cleaning.
    """

    def __init__(self, debug=False):
        """
        Initialize the FeatureLoader.

        Args:
            debug (bool): If True, uses a smaller subset of data for debugging.
        """
        self.debug = debug
        self.valid_columns = None

    def _drop_constant_columns(self, df, fit=False):
        """
        Removes columns that have a single unique value (constant features).

        Args:
            df (pd.DataFrame): Feature matrix.
            fit (bool): If True, identifies constant columns from this dataframe
                        and stores the list of valid columns.

        Returns:
            pd.DataFrame: Feature matrix with constant columns removed.
        """
        if fit:
            # Identify columns with more than 1 unique value
            # We filter for columns present in the dataframe
            keep_cols = [c for c in df.columns if df[c].nunique() > 1]
            self.valid_columns = keep_cols
            return df[keep_cols]
        else:
            if self.valid_columns is None:
                # If not fitted, return original df but warn
                print(
                    "Warning: FeatureLoader not fitted on training data. Skipping constant column removal."
                )
                return df

            # Select only the columns that were valid in the training set
            # Use intersection to avoid KeyErrors if a column is missing in test (unlikely but safe)
            valid_existing_cols = [c for c in self.valid_columns if c in df.columns]
            return df[valid_existing_cols]

    def load_train_val(self, load_cached_data=True):
        """
        Loads the training and validation datasets.

        Args:
            load_cached_data (bool): Whether to load from parquet cache if available.

        Returns:
            X_train (pd.DataFrame): Training features.
            y_train_log (pd.DataFrame): Log-transformed training targets.
            X_val (pd.DataFrame): Validation features.
            y_val_log (pd.DataFrame): Log-transformed validation targets.
        """
        # Load raw features using the library function (handles caching internally)
        train_df = generate_features(
            "train", load_cached_data=load_cached_data, debug=self.debug
        )
        val_df = generate_features(
            "val", load_cached_data=load_cached_data, debug=self.debug
        )

        # Extract Targets
        y_train = train_df[TARGET_COLS]
        y_val = val_df[TARGET_COLS]

        # Apply Log Transformation to Targets
        y_train_log = log_transform(y_train)
        y_val_log = log_transform(y_val)

        # Prepare Feature Matrices (Drop IDs and Targets)
        cols_to_drop = TARGET_COLS + ["id"]
        X_train = train_df.drop(
            columns=[c for c in cols_to_drop if c in train_df.columns]
        )
        X_val = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])

        # Feature Cleaning: Remove constant columns based on Training set statistics
        X_train = self._drop_constant_columns(X_train, fit=True)
        X_val = self._drop_constant_columns(X_val, fit=False)

        return X_train, y_train_log, X_val, y_val_log

    def load_test(self, load_cached_data=True):
        """
        Loads the test dataset.

        Args:
            load_cached_data (bool): Whether to load from parquet cache if available.

        Returns:
            X_test (pd.DataFrame): Test features (cleaned).
            ids_test (pd.Series): IDs corresponding to the test samples.
        """
        # Load raw features
        test_df = generate_features(
            "test", load_cached_data=load_cached_data, debug=self.debug
        )

        # Extract IDs
        ids_test = test_df["id"]

        # Prepare Feature Matrix
        cols_to_drop = TARGET_COLS + ["id"]
        X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

        # Apply Feature Cleaning (using state from training fit)
        X_test = self._drop_constant_columns(X_test, fit=False)

        return X_test, ids_test
