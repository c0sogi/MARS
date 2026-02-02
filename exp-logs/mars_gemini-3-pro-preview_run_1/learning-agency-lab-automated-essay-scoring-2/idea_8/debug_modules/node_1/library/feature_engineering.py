import os
import re
import pandas as pd
import numpy as np
from library.configuration import Config
from library.utilities import get_logger


class FeatureEngineer:
    """
    Responsible for extracting explicit statistical meta-features from essay text.
    These features serve as auxiliary inputs for the Level 2 meta-learner.
    """

    def __init__(self):
        """
        Initialize the FeatureEngineer.
        """
        self.logger = get_logger("FeatureEngineer")
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _count_sentences(self, text):
        """
        Robust sentence counting using regex.
        Splits on '.', '!', '?' followed by whitespace or end of string.
        """
        if not isinstance(text, str) or not text:
            return 0
        # Split by sentence terminators
        sentences = re.split(r"[.!?]+(?:\s+|$)", text.strip())
        # Filter out empty strings resulting from split
        return len([s for s in sentences if s.strip()])

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates statistical features for the given dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing a 'full_text' column.

        Returns:
            pd.DataFrame: Dataframe with added feature columns.
        """
        self.logger.info("Starting feature extraction...")

        if "full_text" not in df.columns:
            raise ValueError("Input dataframe must contain 'full_text' column.")

        # Create a copy to avoid SettingWithCopy warnings on the original df
        df_features = df.copy()

        # Ensure text is string and handle NaNs
        df_features["full_text"] = df_features["full_text"].fillna("").astype(str)

        # 1. Character Count
        df_features["char_count"] = df_features["full_text"].apply(len)

        # 2. Word Count (splitting by whitespace)
        df_features["word_count"] = df_features["full_text"].apply(
            lambda x: len(x.split())
        )

        # 3. Sentence Count
        df_features["sentence_count"] = df_features["full_text"].apply(
            self._count_sentences
        )

        # 4. Unique Word Count
        df_features["unique_word_count"] = df_features["full_text"].apply(
            lambda x: len(set(x.split()))
        )

        # 5. Average Word Length (derived)
        # Avoid division by zero
        df_features["avg_word_len"] = df_features["char_count"] / df_features[
            "word_count"
        ].replace(0, 1)

        self.logger.info(
            f"Feature extraction complete. Added columns: "
            f"{['char_count', 'word_count', 'sentence_count', 'unique_word_count', 'avg_word_len']}"
        )

        return df_features

    def process_and_cache(
        self, df: pd.DataFrame, partition_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Orchestrates feature extraction with caching logic.

        Args:
            df (pd.DataFrame): Input dataframe.
            partition_name (str): Identifier for the data split (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Dataframe with extracted features.
        """
        cache_path = os.path.join(self.cache_dir, f"{partition_name}_features.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading cached features for {partition_name} from {cache_path}"
            )
            try:
                df_cached = pd.read_parquet(cache_path)

                # Validation: Check if cached dataframe matches input length
                if len(df_cached) == len(df):
                    return df_cached
                else:
                    self.logger.warning(
                        f"Cached file length ({len(df_cached)}) does not match input dataframe length ({len(df)}). Recomputing."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing.")

        # 2. Compute from scratch
        self.logger.info(f"Computing features for {partition_name}...")
        df_processed = self.extract_features(df)

        # 3. Save to cache
        try:
            self.logger.info(f"Saving features for {partition_name} to {cache_path}")
            df_processed.to_parquet(cache_path, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_path}: {e}")

        return df_processed
