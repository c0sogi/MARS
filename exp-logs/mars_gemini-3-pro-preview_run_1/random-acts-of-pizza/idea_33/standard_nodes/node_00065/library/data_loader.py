import pandas as pd
import numpy as np
import ast
import os
from library.config import Config
from library.utils import set_seed


class DataLoader:
    """
    Handles loading and preprocessing of raw data from metadata CSVs.
    """

    @staticmethod
    def parse_list_column(df, col_name):
        """
        Parses a column containing stringified lists (from CSV) back into actual Python lists.
        """
        if col_name in df.columns:
            # Check if it looks like a stringified list (starts with [) and is of object type
            if df[col_name].dtype == "object":
                try:
                    # Use ast.literal_eval for safe evaluation of string representations of lists
                    # Handle NaNs by treating them as empty lists
                    df[col_name] = (
                        df[col_name]
                        .fillna("[]")
                        .apply(
                            lambda x: (
                                ast.literal_eval(x)
                                if isinstance(x, str) and x.strip().startswith("[")
                                else x
                            )
                        )
                    )
                except (ValueError, SyntaxError) as e:
                    print(
                        f"Warning: Could not parse column '{col_name}' as list. Error: {e}"
                    )
        return df

    @staticmethod
    def load_data(debug_size=Config.DEBUG_SAMPLE_SIZE):
        """
        Loads train, validation, and test datasets from the metadata CSV files.
        Parses known list columns and applies debug sampling if configured.

        Args:
            debug_size (int, optional): Number of rows to sample for debugging.
                                      Defaults to Config.DEBUG_SAMPLE_SIZE.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        set_seed()  # Ensure reproducibility

        print("Loading datasets from metadata...")
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(f"Train data not found at {Config.TRAIN_DATA_PATH}")

        df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(Config.VAL_DATA_PATH)
        df_test = pd.read_csv(Config.TEST_DATA_PATH)

        # List of columns known to be arrays in the original JSON
        # These are stored as strings in CSV and need parsing
        list_cols = ["requester_subreddits_at_request"]

        for col in list_cols:
            df_train = DataLoader.parse_list_column(df_train, col)
            df_val = DataLoader.parse_list_column(df_val, col)
            df_test = DataLoader.parse_list_column(df_test, col)

        # Apply debug sampling if requested
        if debug_size is not None:
            print(f"Debug Mode: Sampling first {debug_size} rows per dataset.")
            df_train = df_train.head(debug_size)
            df_val = df_val.head(debug_size)
            df_test = df_test.head(debug_size)

        print(
            f"Data Loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
        )
        return df_train, df_val, df_test

    @staticmethod
    def get_feature_intersection(
        df_train, df_test, target_col="requester_received_pizza", exclude_cols=None
    ):
        """
        Identifies the intersection of columns between training and test datasets to ensure
        consistency and prevent leakage. Removes target column and identifiers.

        Args:
            df_train (pd.DataFrame): Training dataframe.
            df_test (pd.DataFrame): Test dataframe.
            target_col (str): Name of the target column to exclude.
            exclude_cols (list, optional): Additional list of columns to exclude.

        Returns:
            list: Sorted list of valid feature column names.
        """
        if exclude_cols is None:
            exclude_cols = []

        # Standard columns to exclude (identifiers, leakage, metadata artifacts)
        # We keep timestamps as they might be used for feature engineering,
        # but exclude obvious IDs and leakage.
        default_excludes = ["request_id", "source_file", "giver_username_if_known"]

        # Combine all exclusions
        all_excludes = set(exclude_cols + default_excludes + [target_col])

        # Get intersection
        train_cols = set(df_train.columns)
        test_cols = set(df_test.columns)
        common_cols = train_cols.intersection(test_cols)

        # Filter and sort
        final_cols = [c for c in common_cols if c not in all_excludes]
        final_cols.sort()

        return final_cols
