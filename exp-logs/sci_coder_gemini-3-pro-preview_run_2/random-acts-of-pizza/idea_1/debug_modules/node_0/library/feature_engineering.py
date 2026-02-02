import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer

from library.config import (
    TEXT_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    HASH_VECTOR_SIZE,
    TRAIN_FEATURES_CACHE,
    VAL_FEATURES_CACHE,
    TEST_FEATURES_CACHE,
    WORKING_DIR,
)
from library.data_loader import load_and_merge_data


class FeaturePreprocessor:
    """
    A class to handle feature engineering:
    1. Text Hashing: Combines title and text, applies HashingVectorizer.
    2. Numeric Imputation: Fills missing values in numerical columns.
    """

    def __init__(self):
        # HashingVectorizer is stateless and memory efficient.
        # alternate_sign=False produces non-negative values.
        # norm=None preserves the magnitude (length information).
        self.vectorizer = HashingVectorizer(
            n_features=HASH_VECTOR_SIZE, alternate_sign=False, norm=None
        )
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols = NUMERIC_COLS
        self.text_cols = TEXT_COLS

    def _get_text_data(self, df):
        """
        Concatenates text columns into a single Series of strings.
        """
        if not self.text_cols:
            return pd.Series([""] * len(df), index=df.index)

        # Start with the first column
        combined_text = df[self.text_cols[0]].fillna("").astype(str)

        # Append subsequent columns
        for col in self.text_cols[1:]:
            combined_text = combined_text + " " + df[col].fillna("").astype(str)

        return combined_text

    def fit(self, df):
        """
        Fits the imputer on the provided DataFrame.
        """
        # Fit imputer on valid numeric columns
        valid_numeric = [c for c in self.numeric_cols if c in df.columns]
        if valid_numeric:
            self.imputer.fit(df[valid_numeric])
        return self

    def transform(self, df):
        """
        Transforms the DataFrame into a feature matrix.
        """
        # 1. Text Processing
        text_data = self._get_text_data(df)
        # Transform text to sparse matrix
        text_features_sparse = self.vectorizer.transform(text_data)
        # Convert to dense array (safe due to small HASH_VECTOR_SIZE)
        text_features = text_features_sparse.toarray()

        # Create DataFrame for text features
        text_feat_cols = [f"text_hash_{i}" for i in range(text_features.shape[1])]
        df_text = pd.DataFrame(text_features, columns=text_feat_cols, index=df.index)

        # 2. Numeric Processing
        valid_numeric = [c for c in self.numeric_cols if c in df.columns]
        if valid_numeric:
            numeric_data = self.imputer.transform(df[valid_numeric])
            df_numeric = pd.DataFrame(
                numeric_data, columns=valid_numeric, index=df.index
            )
        else:
            df_numeric = pd.DataFrame(index=df.index)

        # 3. Concatenate
        X = pd.concat([df_numeric, df_text], axis=1)
        return X


def get_processed_data(load_cached_data=True, debug=False):
    """
    Loads raw data, processes it into features, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
        debug (bool): If True, runs on a subset of data and does not save to cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data and not debug:
        if (
            os.path.exists(TRAIN_FEATURES_CACHE)
            and os.path.exists(VAL_FEATURES_CACHE)
            and os.path.exists(TEST_FEATURES_CACHE)
        ):

            print("Loading features from cache...")
            train_df = pd.read_parquet(TRAIN_FEATURES_CACHE)
            val_df = pd.read_parquet(VAL_FEATURES_CACHE)
            test_df = pd.read_parquet(TEST_FEATURES_CACHE)

            # Separate Features and Targets
            y_train = train_df[TARGET_COL]
            X_train = train_df.drop(columns=[TARGET_COL, "request_id"], errors="ignore")

            y_val = val_df[TARGET_COL]
            X_val = val_df.drop(columns=[TARGET_COL, "request_id"], errors="ignore")

            test_ids = test_df["request_id"]
            X_test = test_df.drop(columns=["request_id"], errors="ignore")

            return X_train, y_train, X_val, y_val, X_test, test_ids

    # 2. Compute from Scratch
    print("Computing features from scratch...")

    # Load raw data
    raw_train, raw_val, raw_test = load_and_merge_data(debug=debug)

    # Initialize and fit preprocessor
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(raw_train)

    # Transform datasets
    X_train = preprocessor.transform(raw_train)
    X_val = preprocessor.transform(raw_val)
    X_test = preprocessor.transform(raw_test)

    # Extract Targets and IDs
    y_train = raw_train[TARGET_COL].astype(int)
    y_val = raw_val[TARGET_COL].astype(int)
    test_ids = raw_test["request_id"]

    # 3. Save to Cache (if not debugging)
    if not debug:
        print(f"Saving features to {WORKING_DIR}...")

        # Prepare DataFrames for saving (include ID and Target)
        train_cache = X_train.copy()
        train_cache[TARGET_COL] = y_train
        train_cache["request_id"] = raw_train["request_id"].values

        val_cache = X_val.copy()
        val_cache[TARGET_COL] = y_val
        val_cache["request_id"] = raw_val["request_id"].values

        test_cache = X_test.copy()
        test_cache["request_id"] = test_ids.values

        # Save
        train_cache.to_parquet(TRAIN_FEATURES_CACHE, index=False)
        val_cache.to_parquet(VAL_FEATURES_CACHE, index=False)
        test_cache.to_parquet(TEST_FEATURES_CACHE, index=False)

    return X_train, y_train, X_val, y_val, X_test, test_ids
