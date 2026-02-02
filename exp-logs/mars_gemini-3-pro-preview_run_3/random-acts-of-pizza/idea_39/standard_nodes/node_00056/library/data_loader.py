import os
import logging
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import timer


class DataLoader:
    """
    Responsible for loading, cleaning, and preprocessing the dataset.
    Implements caching to speed up iterative development.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def clean_text(self, df: pd.DataFrame) -> pd.Series:
        """
        Combines title and body text into a single string.
        Handles missing values by filling them with empty strings.
        """
        title = df["request_title"].fillna("").astype(str)
        # Use edit_aware text to avoid leakage from edits saying "Thanks for pizza"
        body = df["request_text_edit_aware"].fillna("").astype(str)

        # Concatenate with a space separator
        text_combined = title + " " + body
        return text_combined

    def process_subreddits(self, df: pd.DataFrame) -> pd.Series:
        """
        Converts the list of subreddits into a space-separated string
        suitable for TF-IDF vectorization (Bag-of-Concepts).
        """

        def join_subreddits(sub_list):
            if isinstance(sub_list, (list, np.ndarray)):
                return " ".join([str(s) for s in sub_list])
            return ""

        if "requester_subreddits_at_request" in df.columns:
            return df["requester_subreddits_at_request"].apply(join_subreddits)
        else:
            return pd.Series([""] * len(df), index=df.index)

    def extract_metadata(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """
        Selects allow-listed numerical columns and drops leakage columns.
        """
        # Identify columns to drop
        cols_to_drop = set(Config.EXCLUDE_COLS)

        # Identify leakage columns (stats at retrieval time)
        leakage_cols = [c for c in df.columns if str(c).endswith("_at_retrieval")]
        cols_to_drop.update(leakage_cols)

        # Determine features to keep
        # We want numerical columns that are NOT in the drop list
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        features = [c for c in numeric_cols if c not in cols_to_drop]

        # Start with the selected features
        meta_df = df[features].copy()

        # Fill NaNs in numerical columns with 0 (safe default for counts/days)
        # A more complex pipeline might use a fitted imputer, but for the loader
        # we ensure structural integrity.
        meta_df = meta_df.fillna(0)

        # Explicitly add target if it exists and is not 'test'
        if split != "test" and Config.TARGET_COL in df.columns:
            meta_df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(int)

        # Explicitly add ID
        if Config.ID_COL in df.columns:
            meta_df[Config.ID_COL] = df[Config.ID_COL]

        return meta_df

    def load_data(
        self, split: str = "train", load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads the dataset for the specified split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from local cache.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_cleaned.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached {split} data from {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Load raw metadata
        self.logger.info(f"Processing {split} data from metadata...")
        if split == "train":
            path = Config.TRAIN_PATH
        elif split == "val":
            path = Config.VAL_PATH
        elif split == "test":
            path = Config.TEST_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        raw_df = pd.read_parquet(path)

        # 3. Process Data
        with timer(f"Cleaning {split} data"):
            # A. Text Processing
            text_col = self.clean_text(raw_df)

            # B. Subreddit Processing
            sub_col = self.process_subreddits(raw_df)

            # C. Metadata Extraction
            processed_df = self.extract_metadata(raw_df, split)

            # D. Combine
            processed_df["text_combined"] = text_col
            processed_df["subreddit_text"] = sub_col

        # 4. Save to cache
        self.logger.info(f"Saving {split} data to cache: {cache_path}")
        processed_df.to_parquet(cache_path, index=False)

        return processed_df
