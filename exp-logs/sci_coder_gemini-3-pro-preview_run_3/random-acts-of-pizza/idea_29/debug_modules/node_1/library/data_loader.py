import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    Handles loading, cleaning, and caching of the Random Acts of Pizza dataset.
    """

    def __init__(self):
        self.logger = setup_logger("data_loader")
        self.cache_dir = Config.CACHE_DIR

        # Define cache file paths
        self.train_cache_path = os.path.join(self.cache_dir, "cleaned_train.parquet")
        self.val_cache_path = os.path.join(self.cache_dir, "cleaned_val.parquet")
        self.test_cache_path = os.path.join(self.cache_dir, "cleaned_test.parquet")

    def load_raw_data(self, split: str) -> pd.DataFrame:
        """
        Loads raw data from the metadata directory based on the split.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        self.logger.info(f"Loading raw {split} data from {path}")
        return pd.read_parquet(path)

    def clean_data(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        Performs strict leakage prevention and type casting.

        Args:
            df (pd.DataFrame): The dataframe to clean.
            is_test (bool): Whether this is the test set (target column might be missing).

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        # 1. Leakage Prevention: Drop retrieval-time columns
        retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
        if retrieval_cols:
            self.logger.info(
                f"Dropping {len(retrieval_cols)} leakage columns (suffix '_at_retrieval')"
            )
            df = df.drop(columns=retrieval_cols)

        # 2. Type Casting: Target Variable
        if not is_test and Config.TARGET_COL in df.columns:
            df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(int)

        # 3. Ensure Text Columns are Strings
        text_cols = [Config.TEXT_COL, Config.TITLE_COL]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")

        return df

    def load_dataset(self, load_cached_data: bool = True):
        """
        Main entry point to load train, val, and test datasets.
        Handles caching and debug sampling.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Check if cache exists
        cache_exists = (
            os.path.exists(self.train_cache_path)
            and os.path.exists(self.val_cache_path)
            and os.path.exists(self.test_cache_path)
        )

        if load_cached_data and cache_exists:
            self.logger.info("Loading datasets from cache...")
            try:
                train_df = pd.read_parquet(self.train_cache_path)
                val_df = pd.read_parquet(self.val_cache_path)
                test_df = pd.read_parquet(self.test_cache_path)

                self.logger.info(
                    f"Loaded train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
                )
                return train_df, val_df, test_df
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Re-processing data.")

        # Process from scratch
        self.logger.info("Processing data from scratch...")

        # Load Raw
        train_df = self.load_raw_data("train")
        val_df = self.load_raw_data("val")
        test_df = self.load_raw_data("test")

        # Clean
        train_df = self.clean_data(train_df, is_test=False)
        val_df = self.clean_data(val_df, is_test=False)
        test_df = self.clean_data(test_df, is_test=True)

        # Debug Sampling
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Save to Cache
        self.logger.info("Saving processed datasets to cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        train_df.to_parquet(self.train_cache_path, index=False)
        val_df.to_parquet(self.val_cache_path, index=False)
        test_df.to_parquet(self.test_cache_path, index=False)

        self.logger.info(
            f"Final shapes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
        )
        return train_df, val_df, test_df


def load_and_clean_data(load_cached_data: bool = True):
    """
    Wrapper function to instantiate DataLoader and load data.
    """
    loader = DataLoader()
    return loader.load_dataset(load_cached_data=load_cached_data)
