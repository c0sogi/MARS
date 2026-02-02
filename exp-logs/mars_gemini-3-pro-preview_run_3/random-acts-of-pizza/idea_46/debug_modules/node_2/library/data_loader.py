import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_data, save_cache, load_cache, log_info, timer


class DataLoader:
    """
    Handles loading and preprocessing of the RAOP dataset.
    Implements leakage prevention, feature formatting, and caching.
    """

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies structural preprocessing to the raw DataFrame.
        1. Text Concatenation (Title + Edit-Aware Body).
        2. Subreddit List to String conversion.
        3. Feature Selection (Dense Metadata).
        4. Target and ID retention.
        5. Basic Imputation (Fill NaNs with 0 for numericals).

        Args:
            df (pd.DataFrame): Raw DataFrame loaded from metadata.

        Returns:
            pd.DataFrame: Processed DataFrame ready for feature engineering.
        """
        # 1. Text Concatenation
        # Strictly use edit_aware text to prevent leakage from post-hoc edits
        title = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        body = df[Config.TEXT_COL_BODY].fillna("").astype(str)
        df["text_full"] = title + " " + body

        # 2. Subreddit History
        # Convert list of strings to space-separated string for TF-IDF vectorization
        if Config.SUBREDDIT_COL in df.columns:

            def join_subreddits(x):
                if isinstance(x, (list, np.ndarray)):
                    return " ".join(x)
                return str(x) if pd.notnull(x) else ""

            df["subreddit_text"] = df[Config.SUBREDDIT_COL].apply(join_subreddits)
        else:
            df["subreddit_text"] = ""

        # 3. Select Columns
        # We need ID, Target (if exists), Text, Subreddits, and Allow-listed Dense Metadata
        keep_cols = [
            Config.ID_COL,
            "text_full",
            "subreddit_text",
        ] + Config.METADATA_DENSE_FEATURES

        if Config.TARGET_COL in df.columns:
            keep_cols.append(Config.TARGET_COL)

        # 4. Basic Imputation for Dense Features
        # Ensure all requested dense features exist and are filled.
        # While analysis showed no missing values in these specific columns,
        # this is a safeguard for model stability (e.g. Random Forest).
        for col in Config.METADATA_DENSE_FEATURES:
            if col not in df.columns:
                log_info(
                    f"Warning: Dense feature {col} missing in raw data. Filling with 0."
                )
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # Filter DataFrame to keep only relevant columns
        # Use intersection to handle cases where target might not be in test set
        final_cols = [c for c in keep_cols if c in df.columns]
        df_processed = df[final_cols].copy()

        return df_processed

    def load_dataset(self, split: str, load_from_cache: bool = True) -> pd.DataFrame:
        """
        Loads the dataset for a specific split (train, val, test).
        Checks cache first. If not found, loads raw data, processes it, and caches the result.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_from_cache (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        cache_filename = f"processed_data_{split}.parquet"

        # 1. Try Loading from Cache
        if load_from_cache:
            cached_df = load_cache(cache_filename, use_parquet=True)
            if cached_df is not None:
                log_info(f"Loaded {split} data from cache.")
                return cached_df

        # 2. Load Raw and Process
        # load_data handles debug sampling internally based on Config
        with timer(f"Loading and processing {split} data"):
            raw_df = load_data(split)
            processed_df = self.preprocess_data(raw_df)

            # 3. Save to Cache
            save_cache(processed_df, cache_filename, use_parquet=True)

        return processed_df
