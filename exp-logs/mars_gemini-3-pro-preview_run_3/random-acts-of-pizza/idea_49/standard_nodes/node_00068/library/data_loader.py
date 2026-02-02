import os
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    """
    Handles data ingestion, strict leakage prevention, and preprocessing
    for the Hex-View Stacking Ensemble.
    """

    def __init__(self):
        self.train_path = Config.TRAIN_DATA_PATH
        self.val_path = Config.VAL_DATA_PATH
        self.test_path = Config.TEST_DATA_PATH
        self.cache_dir = Config.WORKING_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_raw_data(self):
        """
        Loads the raw stratified metadata files.
        """
        print(f"Loading raw data from {Config.METADATA_DIR}...")
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)
        test_df = pd.read_parquet(self.test_path)
        return train_df, val_df, test_df

    def _process_text_fields(self, df):
        """
        Enforces leakage prevention by using edit-aware text and concatenating
        it with the title.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: DataFrame with a new 'text_combined' column.
        """
        # Fill NaNs with empty strings to allow concatenation
        title = df[Config.TITLE_COL].fillna("").astype(str)
        body = df[Config.TEXT_COL].fillna("").astype(str)

        # Concatenate Title + Body
        df["text_combined"] = title + " " + body
        return df

    def _process_subreddit_history(self, df):
        """
        Converts the list of subreddits into a space-separated string
        suitable for TF-IDF vectorization (Bag-of-Concepts).

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: DataFrame with a new 'subreddit_text' column.
        """
        col = Config.SUBREDDIT_COL

        if col not in df.columns:
            # If column is missing, create empty string column
            df["subreddit_text"] = ""
            return df

        # Function to join list items
        def join_subreddits(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join([str(s) for s in x])
            return ""

        df["subreddit_text"] = df[col].apply(join_subreddits)
        return df

    def get_processed_data(self, load_cached_data=True):
        """
        Main entry point to get processed train, validation, and test sets.
        Implements caching to avoid re-processing.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        train_cache_path = os.path.join(self.cache_dir, "train_processed.parquet")
        val_cache_path = os.path.join(self.cache_dir, "val_processed.parquet")
        test_cache_path = os.path.join(self.cache_dir, "test_processed.parquet")

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):

                print(f"Loading processed data from cache: {self.cache_dir}")
                try:
                    train_df = pd.read_parquet(train_cache_path)
                    val_df = pd.read_parquet(val_cache_path)
                    test_df = pd.read_parquet(test_cache_path)
                    return train_df, val_df, test_df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Re-processing...")
            else:
                print("Cache not found. Processing from scratch...")
        else:
            print("Skipping cache. Processing from scratch...")

        # 2. Process from scratch
        train_df, val_df, test_df = self.load_raw_data()

        # Apply transformations
        print("Processing text fields (Leakage Prevention)...")
        train_df = self._process_text_fields(train_df)
        val_df = self._process_text_fields(val_df)
        test_df = self._process_text_fields(test_df)

        print("Processing subreddit history (Bag-of-Concepts)...")
        train_df = self._process_subreddit_history(train_df)
        val_df = self._process_subreddit_history(val_df)
        test_df = self._process_subreddit_history(test_df)

        # 3. Save to cache
        print(f"Saving processed data to cache: {self.cache_dir}")
        try:
            train_df.to_parquet(train_cache_path, index=False)
            val_df.to_parquet(val_cache_path, index=False)
            test_df.to_parquet(test_cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return train_df, val_df, test_df
