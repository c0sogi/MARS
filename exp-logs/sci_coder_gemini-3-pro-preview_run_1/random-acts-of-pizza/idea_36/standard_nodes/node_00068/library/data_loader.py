import os
import pandas as pd
import numpy as np
import ast
from library.config import Config
from library.utils import save_file, load_file


class DataLoader:
    """
    Handles loading, cleaning, and basic preprocessing of the RAOP dataset.
    Implements caching and leakage prevention.
    """

    @staticmethod
    def parse_list_column(val):
        """
        Parses a string representation of a list back into a list.
        Safe evaluation using ast.literal_eval.
        """
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def get_common_features(df_train, df_test, target_col="requester_received_pizza"):
        """
        Identifies the intersection of columns between train and test sets.
        Ensures strict leakage prevention by removing columns not present in test.
        Preserves the target column in the training set.
        """
        train_cols = set(df_train.columns)
        test_cols = set(df_test.columns)

        # Intersection of features
        common_cols = train_cols.intersection(test_cols)

        # Convert to list and sort for reproducibility
        common_cols = sorted(list(common_cols))

        # Filter Test columns
        df_test_filtered = df_test[common_cols].copy()

        # Filter Train columns (Common + Target)
        train_cols_to_keep = common_cols.copy()
        if target_col in df_train.columns:
            if target_col not in train_cols_to_keep:
                train_cols_to_keep.append(target_col)

        df_train_filtered = df_train[train_cols_to_keep].copy()

        return df_train_filtered, df_test_filtered

    @classmethod
    def load_data(cls, load_cached_data=True):
        """
        Main entry point to load train, val, and test data.

        Args:
            load_cached_data (bool): If True, attempts to load processed parquet files from cache.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        # Define cache paths
        cache_train = os.path.join(Config.WORKING_DIR, "train_cleaned.parquet")
        cache_val = os.path.join(Config.WORKING_DIR, "val_cleaned.parquet")
        cache_test = os.path.join(Config.WORKING_DIR, "test_cleaned.parquet")

        # 1. Try Loading from Cache
        if load_cached_data:
            df_train = load_file(cache_train)
            df_val = load_file(cache_val)
            df_test = load_file(cache_test)

            if df_train is not None and df_val is not None and df_test is not None:
                print("Loaded cleaned data from cache.")
                return df_train, df_val, df_test

        print("Loading raw data from metadata CSVs...")

        # 2. Load Raw Metadata CSVs
        if not os.path.exists(Config.TRAIN_PATH):
            raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_PATH}")

        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # 3. Parse List Columns
        # The CSV format flattens lists to strings (e.g., "['a', 'b']"). We need to parse them.
        list_cols = ["requester_subreddits_at_request"]

        print("Parsing list columns...")
        for col in list_cols:
            if col in df_train.columns:
                df_train[col] = df_train[col].apply(cls.parse_list_column)
            if col in df_val.columns:
                df_val[col] = df_val[col].apply(cls.parse_list_column)
            if col in df_test.columns:
                df_test[col] = df_test[col].apply(cls.parse_list_column)

        # 4. Leakage Prevention
        # Ensure Train and Val have consistent features with Test
        print("Applying leakage prevention (feature intersection)...")
        # Align Train with Test
        df_train, df_test_aligned = cls.get_common_features(df_train, df_test)
        # Align Val with Test (using the same logic)
        df_val, _ = cls.get_common_features(df_val, df_test)
        # Update Test to the aligned version
        df_test = df_test_aligned

        # 5. Debug Sampling
        if Config.DEBUG:
            print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

        # 6. Save to Cache
        print("Saving cleaned data to cache...")
        save_file(df_train, cache_train)
        save_file(df_val, cache_val)
        save_file(df_test, cache_test)

        return df_train, df_val, df_test
