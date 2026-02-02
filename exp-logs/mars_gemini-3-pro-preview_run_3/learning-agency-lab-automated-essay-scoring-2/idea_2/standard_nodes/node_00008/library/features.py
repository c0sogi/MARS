import os
import re
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    """
    Extracts structural and statistical meta-features from essay texts.
    Features include length metrics, sentence counts, and lexical diversity.
    """

    def __init__(self):
        """
        Initialize the FeatureEngineer.
        Sets the random seed for reproducibility.
        """
        seed_everything()

    def _count_sentences(self, text):
        """Approximates sentence count based on punctuation."""
        if not text:
            return 0
        # Split by ., !, ? followed by space or end of string
        sentences = re.split(r"[.!?]+(?:\s+|$)", text)
        # Filter out empty strings resulting from split
        return len([s for s in sentences if s.strip()])

    def _count_paragraphs(self, text):
        """Counts paragraphs based on newlines."""
        if not text:
            return 0
        # Split by one or more newlines
        parts = re.split(r"\n+", text.strip())
        return len([p for p in parts if p.strip()])

    def _extract_single_row_features(self, text):
        """
        Calculates features for a single text entry.

        Args:
            text (str): The essay text.

        Returns:
            dict: A dictionary of calculated features.
        """
        # Handle non-string inputs gracefully
        text = str(text) if text is not None else ""

        # 1. Basic Lengths
        char_count = len(text)

        # Tokenize words (simple regex for alphanumeric sequences)
        words = re.findall(r"\w+", text.lower())
        word_count = len(words)

        # 2. Structural Counts
        sentence_count = self._count_sentences(text)
        paragraph_count = self._count_paragraphs(text)

        # 3. Derived Statistics
        unique_word_count = len(set(words))

        # Avoid division by zero
        avg_word_len = char_count / word_count if word_count > 0 else 0.0
        avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
        lexical_diversity = unique_word_count / word_count if word_count > 0 else 0.0

        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_len": avg_word_len,
            "avg_sentence_len": avg_sentence_len,
            "unique_word_count": unique_word_count,
            "lexical_diversity": lexical_diversity,
        }

    def extract_features(self, df):
        """
        Applies feature extraction to a DataFrame containing a 'full_text' column.

        Args:
            df (pd.DataFrame): Input dataframe with 'full_text'.

        Returns:
            pd.DataFrame: DataFrame containing only the extracted features.
        """
        print("Extracting meta-features...")

        # Apply extraction row-wise
        features_list = (
            df["full_text"].apply(self._extract_single_row_features).tolist()
        )

        # Convert list of dicts to DataFrame
        features_df = pd.DataFrame(features_list)

        # Ensure indices match
        features_df.index = df.index

        return features_df

    def process_split(self, input_path, output_path, load_cached_data=True):
        """
        Orchestrates the loading, processing, and caching of data for a specific split.

        Args:
            input_path (str): Path to the input metadata CSV.
            output_path (str): Path where the feature Parquet file should be saved.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The computed or loaded feature dataframe.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached meta-features from {output_path}")
            try:
                return pd.read_parquet(output_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Computing meta-features for {input_path}...")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)

        # If debugging, sample the data
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLES)

        features_df = self.extract_features(df)

        # 3. Save to cache
        print(f"Saving meta-features to {output_path}")
        features_df.to_parquet(output_path, index=False)

        return features_df

    def run(self, load_cached_data=True):
        """
        Main entry point to process Train, Validation, and Test sets.

        Args:
            load_cached_data (bool): Whether to use cached data if available.
        """
        print("Starting Meta-Feature Engineering...")

        # Process Train
        self.process_split(
            input_path=Config.TRAIN_DATA_PATH,
            output_path=Config.TRAIN_META_FEATS_PATH,
            load_cached_data=load_cached_data,
        )

        # Process Validation
        self.process_split(
            input_path=Config.VAL_DATA_PATH,
            output_path=Config.VAL_META_FEATS_PATH,
            load_cached_data=load_cached_data,
        )

        # Process Test
        self.process_split(
            input_path=Config.TEST_DATA_PATH,
            output_path=Config.TEST_META_FEATS_PATH,
            load_cached_data=load_cached_data,
        )

        print("Meta-Feature Engineering Complete.")
