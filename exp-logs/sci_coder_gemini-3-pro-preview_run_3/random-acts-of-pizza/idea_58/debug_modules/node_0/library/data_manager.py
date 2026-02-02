import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed


class DataManager:
    def __init__(self):
        """
        Initializes the DataManager with paths and configuration from the library.
        """
        self.train_path = Config.TRAIN_PATH
        self.val_path = Config.VAL_PATH
        self.test_path = Config.TEST_PATH
        self.working_dir = Config.WORKING_DIR
        self.meta_features = Config.META_FEATURES_ALLOWLIST

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def load_union_data(self, load_cached_data: bool = True):
        """
        Loads the union of training and validation data, and the test data.
        Applies processing, cleaning, and feature engineering.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.

        Returns:
            tuple: (train_union_df, test_df)
        """
        return self.get_processed_data(load_cached_data=load_cached_data)

    def get_processed_data(self, load_cached_data: bool = True):
        """
        Manages the caching and processing of data.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.

        Returns:
            tuple: (train_processed_df, test_processed_df)
        """
        train_cache_path = os.path.join(
            self.working_dir, "train_union_processed.parquet"
        )
        test_cache_path = os.path.join(self.working_dir, "test_processed.parquet")

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
                try:
                    train_df = pd.read_parquet(train_cache_path)
                    test_df = pd.read_parquet(test_cache_path)
                    return train_df, test_df
                except Exception:
                    # If load fails, proceed to process from scratch
                    pass

        # 2. Process from scratch
        # Load raw metadata
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path):
            raise FileNotFoundError(
                "Metadata files not found. Please ensure ./metadata/ contains train.parquet and val.parquet"
            )

        train_raw = pd.read_parquet(self.train_path)
        val_raw = pd.read_parquet(self.val_path)
        test_raw = pd.read_parquet(self.test_path)

        # Merge Train and Val into Union Dataset
        train_union = pd.concat([train_raw, val_raw], axis=0, ignore_index=True)

        # Process datasets
        train_processed, test_processed = self._process_datasets(train_union, test_raw)

        # Save to cache
        train_processed.to_parquet(train_cache_path, index=False)
        test_processed.to_parquet(test_cache_path, index=False)

        return train_processed, test_processed

    def _process_datasets(self, train_df, test_df):
        """
        Internal method to apply transformations to both datasets consistently.

        Args:
            train_df (pd.DataFrame): Raw union training data.
            test_df (pd.DataFrame): Raw test data.

        Returns:
            tuple: (processed_train, processed_test)
        """
        # 1. Text Processing (Concatenation)
        train_df = self._generate_text_features(train_df)
        test_df = self._generate_text_features(test_df)

        # 2. Subreddit List Processing (List to String for Vectorizer)
        train_df = self._process_subreddits(train_df)
        test_df = self._process_subreddits(test_df)

        # 3. Metadata Selection & Imputation
        # We calculate median on the training set and apply to both to prevent leakage
        medians = train_df[self.meta_features].median()

        train_df = self._impute_metadata(train_df, medians)
        test_df = self._impute_metadata(test_df, medians)

        # 4. Filter Columns (Leakage Prevention)
        # We explicitly select only the columns we need, dropping everything else
        # This automatically handles the removal of _at_retrieval columns

        # Base columns needed for all models
        keep_cols_base = [
            "request_id",
            "text_combined",
            "subreddit_string",
        ] + self.meta_features

        # Train has target, Test does not
        train_cols = keep_cols_base + ["requester_received_pizza"]
        test_cols = keep_cols_base

        # Create final dataframes with strictly typed copies
        train_processed = train_df[train_cols].copy()
        test_processed = test_df[test_cols].copy()

        return train_processed, test_processed

    def _generate_text_features(self, df):
        """
        Concatenates title and edit-aware text into a single column.
        """
        # Fill NaNs with empty string to ensure string concatenation works
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)

        # Concatenate with a space separator
        df["text_combined"] = title + " " + body
        return df

    def _process_subreddits(self, df):
        """
        Converts list of subreddits to space-separated string for Bag-of-Words processing.
        """
        if "requester_subreddits_at_request" in df.columns:

            def join_subs(x):
                if isinstance(x, (list, np.ndarray)):
                    # Filter out None/NaN values inside the list just in case
                    valid_subs = [str(s) for s in x if s is not None]
                    return " ".join(valid_subs)
                return str(x) if pd.notnull(x) else ""

            df["subreddit_string"] = df["requester_subreddits_at_request"].apply(
                join_subs
            )
        else:
            # Fallback if column is missing
            df["subreddit_string"] = ""
        return df

    def _impute_metadata(self, df, medians):
        """
        Imputes allow-listed metadata columns with provided medians.
        """
        for col in self.meta_features:
            if col in df.columns:
                df[col] = df[col].fillna(medians[col])
            else:
                # If a column from allowlist is completely missing in df, fill with median
                # (This handles cases where a feature might be missing from raw data)
                df[col] = medians.get(col, 0)
        return df
