import os
import ast
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


class DataLoader:
    """
    Handles loading of raw data from metadata CSVs, preprocessing of list columns,
    and identification of safe tabular features to prevent leakage.
    """

    @staticmethod
    def load_raw_data(load_cached_data=True):
        """
        Loads train, validation, and test datasets.

        Logic:
        1. If load_cached_data is True and parquet files exist in Config.WORKING_DIR, load them.
        2. Otherwise, load from metadata CSVs.
        3. Parse stringified list columns (e.g. subreddit lists).
        4. If Config.DEBUG is True, subsample the data.
        5. Save the processed dataframes to parquet in Config.WORKING_DIR for caching.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        set_seed()

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache file paths
        train_cache_path = os.path.join(Config.WORKING_DIR, "train.parquet")
        val_cache_path = os.path.join(Config.WORKING_DIR, "val.parquet")
        test_cache_path = os.path.join(Config.WORKING_DIR, "test.parquet")

        # Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):
                print(f"Loading cached data from {Config.WORKING_DIR}...")
                df_train = pd.read_parquet(train_cache_path)
                df_val = pd.read_parquet(val_cache_path)
                df_test = pd.read_parquet(test_cache_path)
                return df_train, df_val, df_test
            else:
                print("Cache not found or incomplete. Processing from scratch...")

        # Load from metadata CSVs
        print("Loading raw data from metadata...")
        train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")

        # Verify files exist
        for p in [train_csv_path, val_csv_path, test_csv_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Metadata file not found: {p}")

        df_train = pd.read_csv(train_csv_path)
        df_val = pd.read_csv(val_csv_path)
        df_test = pd.read_csv(test_csv_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"DEBUG mode enabled. Subsampling to {Config.DEBUG_SIZE} samples.")
            df_train = df_train.iloc[: Config.DEBUG_SIZE].reset_index(drop=True)
            df_val = df_val.iloc[: Config.DEBUG_SIZE].reset_index(drop=True)
            df_test = df_test.iloc[: Config.DEBUG_SIZE].reset_index(drop=True)

        # Parse list columns (e.g., 'requester_subreddits_at_request')
        # CSV stores lists as string representations "['a', 'b']"
        def parse_list_column(df, col_name):
            if col_name in df.columns:
                # Handle NaNs and ensure string type before parsing
                df[col_name] = df[col_name].fillna("[]").astype(str)
                try:
                    df[col_name] = df[col_name].apply(ast.literal_eval)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Failed to parse list column {col_name}: {e}")
                    # Fallback: return empty list on failure
                    df[col_name] = df[col_name].apply(lambda x: [])
            return df

        print(f"Parsing list column: {Config.SUBREDDIT_LIST_COL}")
        df_train = parse_list_column(df_train, Config.SUBREDDIT_LIST_COL)
        df_val = parse_list_column(df_val, Config.SUBREDDIT_LIST_COL)
        df_test = parse_list_column(df_test, Config.SUBREDDIT_LIST_COL)

        # Save to cache
        print(f"Saving processed data to cache at {Config.WORKING_DIR}...")
        df_train.to_parquet(train_cache_path, index=False)
        df_val.to_parquet(val_cache_path, index=False)
        df_test.to_parquet(test_cache_path, index=False)

        return df_train, df_val, df_test

    @staticmethod
    def filter_leakage_columns(df_train, df_test):
        """
        Determines the safe subset of tabular features by finding the intersection
        of columns in train and test, and removing explicit drop columns (IDs, text, etc.).

        This effectively removes 'at_retrieval' columns present in train but not test.

        Args:
            df_train (pd.DataFrame): Training dataframe.
            df_test (pd.DataFrame): Test dataframe.

        Returns:
            list: Sorted list of safe feature column names.
        """
        # Find intersection
        train_cols = set(df_train.columns)
        test_cols = set(df_test.columns)
        common_cols = train_cols.intersection(test_cols)

        # Identify columns to exclude
        # Config.DROP_COLS includes IDs, Text, Subreddit Lists, and Target
        exclude_cols = set(Config.DROP_COLS)

        # Ensure target is excluded if it happens to be in common_cols (unlikely for test, but safe)
        if Config.TARGET_COL in common_cols:
            exclude_cols.add(Config.TARGET_COL)

        # Filter
        safe_features = [c for c in common_cols if c not in exclude_cols]
        safe_features.sort()

        print(
            f"Feature Selection: {len(safe_features)} safe tabular features identified."
        )
        return safe_features
