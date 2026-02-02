import os
import pandas as pd
import numpy as np
from library import config, utils


class DataLoader:
    """
    Responsible for loading the raw data splits from the metadata directory.
    Wraps the utility functions to provide a clean interface for the pipeline.
    """

    @staticmethod
    def load_train(sample_size=None):
        """
        Loads the training dataset.

        Args:
            sample_size (int, optional): Number of samples to load.

        Returns:
            pd.DataFrame: Training data.
        """
        return utils.load_dataset("train", sample_size=sample_size)

    @staticmethod
    def load_val(sample_size=None):
        """
        Loads the validation dataset.

        Args:
            sample_size (int, optional): Number of samples to load.

        Returns:
            pd.DataFrame: Validation data.
        """
        return utils.load_dataset("val", sample_size=sample_size)

    @staticmethod
    def load_test(sample_size=None):
        """
        Loads the test dataset.

        Args:
            sample_size (int, optional): Number of samples to load.

        Returns:
            pd.DataFrame: Test data.
        """
        return utils.load_dataset("test", sample_size=sample_size)


class DataCleaner:
    """
    Responsible for initial cleaning and formatting of the datasets.
    Handles type conversions, list-to-string transformations, and basic null handling.
    Implements caching to speed up subsequent runs.
    """

    def __init__(self):
        self.logger = utils.get_logger("DataCleaner")
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def clean_data(self, df, split_name, load_cached_data=True):
        """
        Cleans the provided dataframe or loads a cleaned version from cache.

        Processing steps:
        1. Convert subreddit lists to space-separated strings.
        2. Ensure text columns are strings and fill NaNs.
        3. Fill missing numerical values with 0 (basic safety fill).

        Args:
            df (pd.DataFrame): The raw dataframe to clean.
            split_name (str): The name of the split (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        cache_path = os.path.join(self.cache_dir, f"{split_name}_cleaned.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading cleaned {split_name} data from cache: {cache_path}"
            )
            try:
                cached_df = pd.read_parquet(cache_path)
                # Verify length matches (in case sample_size changed in loader but cache persists)
                if len(cached_df) == len(df):
                    return cached_df
                else:
                    self.logger.info(
                        f"Cache size mismatch ({len(cached_df)} vs {len(df)}). Recomputing..."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        self.logger.info(f"Cleaning {split_name} data...")

        # Create a copy to avoid SettingWithCopy warnings on the original df
        cleaned_df = df.copy()

        # 2. Process History Column (List -> String)
        # The input JSON/Parquet might have this as a list or numpy array of objects
        if config.HISTORY_COL in cleaned_df.columns:
            # Check if the first non-null element is a list/array-like
            first_valid = (
                cleaned_df[config.HISTORY_COL].dropna().iloc[0]
                if not cleaned_df[config.HISTORY_COL].dropna().empty
                else None
            )

            if isinstance(first_valid, (list, np.ndarray)):
                self.logger.info(
                    f"Converting {config.HISTORY_COL} from list to string..."
                )
                cleaned_df[config.HISTORY_COL] = cleaned_df[config.HISTORY_COL].apply(
                    lambda x: (
                        " ".join(x)
                        if isinstance(x, (list, np.ndarray))
                        else str(x) if pd.notnull(x) else ""
                    )
                )
            else:
                # Ensure it is string even if it was already loaded as string
                cleaned_df[config.HISTORY_COL] = (
                    cleaned_df[config.HISTORY_COL].fillna("").astype(str)
                )

        # 3. Process Text Columns
        text_cols = [config.TEXT_COL, config.TITLE_COL]
        for col in text_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna("").astype(str)

        # 4. Basic Numerical Imputation
        # We fill with 0 for safety here. More sophisticated imputation (median)
        # should happen in feature engineering using a fitted imputer to avoid leakage.
        # However, for 'DataCleaner', ensuring no NaNs in numeric columns prevents crashes.
        # We only touch numeric columns that are not excluded or IDs.
        numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns
        exclude_cols = set(config.EXCLUDE_COLS)
        cols_to_fill = [
            c for c in numeric_cols if c not in exclude_cols and c != config.TARGET_COL
        ]

        if cols_to_fill:
            cleaned_df[cols_to_fill] = cleaned_df[cols_to_fill].fillna(0)

        # 5. Save to cache
        self.logger.info(f"Saving cleaned {split_name} data to cache: {cache_path}")
        try:
            cleaned_df.to_parquet(cache_path, index=False)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

        return cleaned_df
