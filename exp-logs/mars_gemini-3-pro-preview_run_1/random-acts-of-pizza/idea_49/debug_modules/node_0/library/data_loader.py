import os
import ast
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    """
    Handles loading of the Pizza Request dataset.
    Uses the stratified metadata CSVs as the source of truth for splits.
    Implements caching to Parquet for efficiency and preserves data types (e.g. lists).
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        Config.ensure_dirs()

        self.train_cache_path = os.path.join(self.cache_dir, "train_base.parquet")
        self.val_cache_path = os.path.join(self.cache_dir, "val_base.parquet")
        self.test_cache_path = os.path.join(self.cache_dir, "test_base.parquet")

    def load_dataset(self, load_cached_data=True):
        """
        Loads the train, validation, and test datasets.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.
                                     If False or cache miss, reloads from CSVs and saves to cache.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        if load_cached_data:
            if (
                os.path.exists(self.train_cache_path)
                and os.path.exists(self.val_cache_path)
                and os.path.exists(self.test_cache_path)
            ):
                print("Loading datasets from cache...")
                try:
                    train_df = pd.read_parquet(self.train_cache_path)
                    val_df = pd.read_parquet(self.val_cache_path)
                    test_df = pd.read_parquet(self.test_cache_path)

                    # Apply debug limits if configured, even on cached data
                    if Config.DEBUG or Config.MAX_SAMPLES:
                        train_df, val_df, test_df = self._apply_limits(
                            train_df, val_df, test_df
                        )

                    return train_df, val_df, test_df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reloading from source.")
            else:
                print("Cache not found. Loading from source CSVs...")
        else:
            print("Force reload requested. Loading from source CSVs...")

        # Load from Metadata CSVs
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Parse stringified lists back to python lists
        # The CSV format saves lists as strings like "['a', 'b']"
        list_cols = ["requester_subreddits_at_request"]

        for df in [train_df, val_df, test_df]:
            for col in list_cols:
                if col in df.columns:
                    # Use literal_eval to safely parse the string representation of lists
                    # Handle NaNs or non-string types gracefully
                    df[col] = df[col].apply(
                        lambda x: (
                            ast.literal_eval(x)
                            if isinstance(x, str) and x.startswith("[")
                            else []
                        )
                    )

        # Apply debug limits before saving to avoid caching truncated data if we want the cache to be full
        # However, usually cache should store full data.
        # But if we are in debug mode, we might not want to wait for full processing.
        # Strategy: Save FULL data to cache, then slice for return.

        print("Saving processed datasets to cache...")
        train_df.to_parquet(self.train_cache_path, index=False)
        val_df.to_parquet(self.val_cache_path, index=False)
        test_df.to_parquet(self.test_cache_path, index=False)

        # Apply limits for current run
        if Config.DEBUG or Config.MAX_SAMPLES:
            train_df, val_df, test_df = self._apply_limits(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _apply_limits(self, train_df, val_df, test_df):
        """Helper to downsample datasets for debugging."""
        limit = Config.MAX_SAMPLES if Config.MAX_SAMPLES else 100
        if Config.DEBUG and Config.MAX_SAMPLES is None:
            limit = 100  # Default debug limit
        elif Config.MAX_SAMPLES:
            limit = Config.MAX_SAMPLES
        else:
            return train_df, val_df, test_df

        print(f"Debug mode/Limit active: Downsampling to {limit} samples per split.")
        return (train_df.head(limit), val_df.head(limit), test_df.head(limit))
