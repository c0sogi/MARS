import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from library.config import Config


class DataProcessor:
    """
    Handles data loading, cleaning, feature engineering, and caching
    for the Pizza Request prediction task.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.train_cache_path = os.path.join(self.cache_dir, "train_processed.parquet")
        self.val_cache_path = os.path.join(self.cache_dir, "val_processed.parquet")
        self.test_cache_path = os.path.join(self.cache_dir, "test_processed.parquet")

    def process_data(self, load_cached_data=True):
        """
        Main method to load and process data.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                os.path.exists(self.train_cache_path)
                and os.path.exists(self.val_cache_path)
                and os.path.exists(self.test_cache_path)
            ):
                print("Loading processed data from cache...")
                train_df = pd.read_parquet(self.train_cache_path)
                val_df = pd.read_parquet(self.val_cache_path)
                test_df = pd.read_parquet(self.test_cache_path)
                return train_df, val_df, test_df

        # 2. Load Raw Data
        print("Loading raw data from metadata...")
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Debug Sampling
        if Config.DEBUG_SAMPLE_SIZE is not None:
            print(f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # 3. Feature Engineering
        print("Engineering features...")
        train_df = self._engineer_features(train_df)
        val_df = self._engineer_features(val_df)
        test_df = self._engineer_features(test_df)

        # Align columns: Ensure Train/Val only use features present in Test
        # This prevents KeyErrors during imputation if Train has features Test lacks (e.g., 'post_was_edited')
        common_cols = [c for c in train_df.columns if c in test_df.columns]

        # Define columns to keep for Train/Val (Common + Target)
        train_cols_to_keep = list(common_cols)
        if (
            Config.TARGET_COL in train_df.columns
            and Config.TARGET_COL not in train_cols_to_keep
        ):
            train_cols_to_keep.append(Config.TARGET_COL)

        # Subset DataFrames
        train_df = train_df[train_cols_to_keep]
        val_df = val_df[train_cols_to_keep]
        test_df = test_df[common_cols]

        # 4. Imputation
        # Identify numerical columns (excluding ID and Target and Text)
        numeric_cols = []
        for col in train_df.columns:
            if col in [Config.ID_COL, Config.TARGET_COL, Config.TEXT_COL]:
                continue
            if pd.api.types.is_numeric_dtype(train_df[col]):
                numeric_cols.append(col)

        print(f"Imputing {len(numeric_cols)} numerical columns...")
        if numeric_cols:
            imputer = SimpleImputer(strategy="median")
            # Fit on train
            train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
            # Transform val and test
            val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])
            test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

        # 5. Save to Cache
        print("Saving processed data to cache...")
        train_df.to_parquet(self.train_cache_path, index=False)
        val_df.to_parquet(self.val_cache_path, index=False)
        test_df.to_parquet(self.test_cache_path, index=False)

        return train_df, val_df, test_df

    def _engineer_features(self, df):
        """
        Applies feature engineering and column selection.
        """
        df = df.copy()

        # Ensure Text Column is clean
        # If the specific edit-aware column is missing (unlikely given schema), fallback to empty string
        if Config.TEXT_COL not in df.columns:
            df[Config.TEXT_COL] = ""
        df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("").astype(str)

        # --- Derived Features ---

        # Time-based
        # Use UTC timestamp if available, else standard
        ts_col = "unix_timestamp_of_request_utc"
        if ts_col not in df.columns:
            ts_col = "unix_timestamp_of_request"

        if ts_col in df.columns:
            # Convert to datetime
            dt = pd.to_datetime(df[ts_col], unit="s")
            df["request_hour"] = dt.dt.hour
            df["request_day_of_week"] = dt.dt.dayofweek
        else:
            # Fallback if timestamps missing
            df["request_hour"] = 0
            df["request_day_of_week"] = 0

        # Word Count
        df["request_word_count"] = df[Config.TEXT_COL].apply(lambda x: len(x.split()))
        df["request_text_len_char"] = df[Config.TEXT_COL].apply(len)

        # --- Column Selection ---

        # Start with mandatory columns
        keep_cols = [Config.ID_COL, Config.TEXT_COL]
        if Config.TARGET_COL in df.columns:
            keep_cols.append(Config.TARGET_COL)

        # Add allow-listed derived features
        # We iterate Config.DERIVED_FEATURES and ensure they are in df
        for feature in Config.DERIVED_FEATURES:
            if feature in df.columns:
                keep_cols.append(feature)

        # Add other safe numerical features
        # Rule: Numeric AND NOT Leakage (ends with _at_retrieval) AND NOT already kept
        for col in df.columns:
            if col in keep_cols:
                continue

            # Check leakage
            if col.endswith(Config.RETRIEVAL_SUFFIX):
                continue

            # Check type
            if pd.api.types.is_numeric_dtype(df[col]):
                keep_cols.append(col)

        # Return subset
        return df[keep_cols]
