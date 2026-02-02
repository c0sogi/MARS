import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from library.config import Config


class DataPreprocessor:
    """
    Handles data loading, feature engineering, scaling, and caching for the DCNv2 pipeline.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Map A-Z to 0-25
        self.char_map = {chr(i + 65): i for i in range(26)}

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering:
        1. Calculates unique character count in f_27.
        2. Decomposes f_27 into 10 separate integer-encoded columns.
        3. Drops original f_27 and source_path.
        """
        # Ensure target is separated or handled, but here we modify in place

        # 1. Unique Character Count
        # We assume f_27 exists.
        if Config.CATEGORICAL_COL in df.columns:
            df["unique_character_count"] = (
                df[Config.CATEGORICAL_COL]
                .apply(lambda x: len(set(x)))
                .astype(np.float32)
            )

            # 2. Decompose String
            # Convert string series to list of lists, then to DataFrame
            # This is generally faster than apply for large datasets
            char_lists = df[Config.CATEGORICAL_COL].apply(list).tolist()
            char_df = pd.DataFrame(char_lists, index=df.index)

            # Rename columns
            char_cols = [f"char_{i}" for i in range(Config.STR_LEN)]
            char_df.columns = char_cols

            # Encode characters to integers
            for col in char_cols:
                char_df[col] = (
                    char_df[col].map(self.char_map).fillna(0).astype(np.int32)
                )

            # Concatenate
            df = pd.concat([df, char_df], axis=1)

            # Drop original categorical column
            df.drop(columns=[Config.CATEGORICAL_COL], inplace=True)

        # Drop source_path if it exists (artifact from metadata generation)
        if "source_path" in df.columns:
            df.drop(columns=["source_path"], inplace=True)

        return df

    def process_data(self, load_cached_data: bool = True, debug: bool = False):
        """
        Main driver for data processing.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.
            debug (bool): If True, limits dataset size for quick debugging.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Define cache paths
        train_cache = Config.TRAIN_PROCESSED_PATH
        val_cache = Config.VAL_PROCESSED_PATH
        test_cache = Config.TEST_PROCESSED_PATH

        cache_exists = (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        )

        # 1. Try Loading Cache
        if load_cached_data and cache_exists:
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            if debug:
                print("Debug mode: Sampling cached data...")
                train_df = train_df.head(10000)
                val_df = val_df.head(2000)
                test_df = test_df.head(2000)

            return train_df, val_df, test_df

        # 2. Process from Scratch
        print("Processing data from scratch...")

        # Load raw data from metadata pointers
        # Note: The metadata files in ./metadata/ contain the actual data splits
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug sampling (before processing to save time)
        if debug:
            print("Debug mode: Sampling raw data...")
            train_df = train_df.head(10000)
            val_df = val_df.head(2000)
            test_df = test_df.head(2000)

        # Apply Feature Engineering
        print("Applying feature engineering...")
        train_df = self._feature_engineering(train_df)
        val_df = self._feature_engineering(val_df)
        test_df = self._feature_engineering(test_df)

        # Identify Continuous Columns for Scaling
        # Base columns + the new engineered count feature
        cont_cols = Config.BASE_CONT_COLS + ["unique_character_count"]

        # Ensure columns exist (safety check)
        cont_cols = [c for c in cont_cols if c in train_df.columns]

        # Fit Scaler (Train only)
        print("Fitting scaler on training data...")
        self.scaler.fit(train_df[cont_cols])

        # Transform
        print("Transforming datasets...")
        train_df[cont_cols] = self.scaler.transform(train_df[cont_cols]).astype(
            np.float32
        )
        val_df[cont_cols] = self.scaler.transform(val_df[cont_cols]).astype(np.float32)
        test_df[cont_cols] = self.scaler.transform(test_df[cont_cols]).astype(
            np.float32
        )

        # Ensure Target Type
        if Config.TARGET_COL in train_df.columns:
            train_df[Config.TARGET_COL] = train_df[Config.TARGET_COL].astype(np.float32)
        if Config.TARGET_COL in val_df.columns:
            val_df[Config.TARGET_COL] = val_df[Config.TARGET_COL].astype(np.float32)

        # 3. Save to Cache (Only if not debugging)
        if not debug:
            print("Saving processed data to cache...")
            Config.create_dirs()
            train_df.to_parquet(train_cache)
            val_df.to_parquet(val_cache)
            test_df.to_parquet(test_cache)

        return train_df, val_df, test_df
