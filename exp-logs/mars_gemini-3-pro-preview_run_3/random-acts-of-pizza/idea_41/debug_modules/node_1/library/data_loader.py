import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    get_logger,
    load_metadata_splits,
    save_cache_parquet,
    load_cache_parquet,
    Timer,
)

logger = get_logger("DataLoader")


class DataPreprocessor:
    """
    Handles initial data cleaning, leakage prevention, and basic preprocessing.
    """

    def __init__(self):
        self.medians = {}
        self.retrieval_suffix = "_at_retrieval"
        self.leakage_cols = ["giver_username_if_known"]
        self.text_cols = (
            Config.TEXT_COLS
        )  # ["request_title", "request_text_edit_aware"]

    def fit(self, df):
        """
        Learns imputation statistics (median) from the training data.
        """
        # Identify numerical columns for imputation
        # We process all numeric columns found in the dataframe to ensure consistency
        # excluding ID and Target columns
        exclude_cols = [Config.ID_COL, Config.TARGET_COL]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_impute = [c for c in numeric_cols if c not in exclude_cols]

        # Calculate medians
        self.medians = df[cols_to_impute].median().to_dict()
        return self

    def transform(self, df):
        """
        Applies transformations: dropping leakage columns, text concatenation, and imputation.
        """
        df = df.copy()

        # 1. Leakage Prevention
        # Drop columns suffixed with _at_retrieval (future info)
        cols_to_drop = [c for c in df.columns if c.endswith(self.retrieval_suffix)]

        # Drop specific leakage columns like giver username
        for col in self.leakage_cols:
            if col in df.columns:
                cols_to_drop.append(col)

        # Drop source_file metadata if present
        if "source_file" in df.columns:
            cols_to_drop.append("source_file")

        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # 2. Text Concatenation
        # Combine title and body into a single text column for NLP models
        # Handle NaNs by replacing with empty string before concatenation
        title = df[self.text_cols[0]].fillna("").astype(str)
        body = df[self.text_cols[1]].fillna("").astype(str)
        df["text_concat"] = title + " " + body

        # 3. Imputation
        # Apply learned medians to missing values
        for col, median_val in self.medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        return df

    def fit_transform(self, df):
        """
        Fits on the dataframe and then transforms it.
        """
        return self.fit(df).transform(df)


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads data from metadata, applies preprocessing, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache miss, re-processes and overwrites cache.

    Returns:
        tuple: (train_df, val_df, test_df) - Preprocessed DataFrames.
    """
    # Define cache filenames
    cache_files = {
        "train": "train_cleaned.parquet",
        "val": "val_cleaned.parquet",
        "test": "test_cleaned.parquet",
    }

    # Attempt to load from cache
    if load_cached_data:
        logger("Checking for cached preprocessed data...")
        train_df = load_cache_parquet(cache_files["train"])
        val_df = load_cache_parquet(cache_files["val"])
        test_df = load_cache_parquet(cache_files["test"])

        if train_df is not None and val_df is not None and test_df is not None:
            logger("Successfully loaded data from cache.")
            return train_df, val_df, test_df
        else:
            logger("Cache miss or incomplete. Proceeding to process from scratch.")

    # Load raw metadata
    with Timer("Load Metadata"):
        logger("Loading raw metadata splits...")
        raw_train, raw_val, raw_test = load_metadata_splits()

    # Preprocess
    with Timer("Preprocess Data"):
        logger("Fitting preprocessor on training data...")
        preprocessor = DataPreprocessor()
        preprocessor.fit(raw_train)

        logger("Transforming datasets...")
        train_clean = preprocessor.transform(raw_train)
        val_clean = preprocessor.transform(raw_val)
        test_clean = preprocessor.transform(raw_test)

    # Save to cache
    with Timer("Save Cache"):
        logger("Saving processed data to cache...")
        save_cache_parquet(train_clean, cache_files["train"])
        save_cache_parquet(val_clean, cache_files["val"])
        save_cache_parquet(test_clean, cache_files["test"])

    return train_clean, val_clean, test_clean
