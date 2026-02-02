import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger
from library.vocab_manager import Vocab

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """
    Handles the extraction of explicit morphological features (Regex) and
    probabilistic features (Priors) from text data. Also manages the
    construction of the Knowledge Base for the hybrid normalization system.
    """

    def __init__(self):
        """
        Initialize the FeatureEngineer and pre-compile regex patterns for efficiency.
        """
        self.regex_feature_dim = Config.NUM_REGEX_FEATURES

        # Pre-compile regex patterns to speed up extraction over large datasets
        self.re_decimal = re.compile(r"^\d+\.\d+$")
        self.re_money = re.compile(r"^[$£€¥]")
        self.re_time = re.compile(r"^\d{1,2}:\d{2}")
        self.re_date = re.compile(r"\d+[/-]\d+[/-]\d+")
        self.re_large_int = re.compile(r"^\d{5,}$")  # Digits only, length > 4

    def _extract_single_token_features(self, token: str) -> List[float]:
        """
        Extracts a fixed-size vector of explicit morphological features for a single token.

        Features (15 total):
        1. is_digit: All characters are digits
        2. is_alpha: All characters are letters
        3. is_alnum: Alphanumeric but not purely digit or alpha
        4. is_upper: All characters are uppercase
        5. is_title: Title case (First char upper, rest lower)
        6. has_digit: Contains at least one digit
        7. is_decimal: Matches digit.digit pattern
        8. is_large_int: Pure digit string longer than 4 chars
        9. has_comma: Contains a comma
        10. has_dot: Contains a period
        11. is_money: Starts with a currency symbol
        12. is_time: Matches time pattern (e.g., 12:30)
        13. is_date: Matches date pattern (e.g., 2020-01-01 or 1/1/20)
        14. is_single: Is a single character
        15. norm_len: Length of token normalized by 20 (capped at 1.0)
        """
        # Basic string checks
        is_digit = 1.0 if token.isdigit() else 0.0
        is_alpha = 1.0 if token.isalpha() else 0.0
        is_alnum = 1.0 if token.isalnum() and not is_digit and not is_alpha else 0.0
        is_upper = 1.0 if token.isupper() else 0.0
        is_title = 1.0 if token.istitle() else 0.0
        has_digit = 1.0 if any(c.isdigit() for c in token) else 0.0

        # Regex and structural checks
        is_decimal = 1.0 if self.re_decimal.match(token) else 0.0
        is_large_int = 1.0 if self.re_large_int.match(token) else 0.0
        has_comma = 1.0 if "," in token else 0.0
        has_dot = 1.0 if "." in token else 0.0
        is_money = 1.0 if self.re_money.match(token) else 0.0
        is_time = 1.0 if self.re_time.match(token) else 0.0
        is_date = 1.0 if self.re_date.search(token) else 0.0
        is_single = 1.0 if len(token) == 1 else 0.0

        # Normalized length
        norm_len = min(len(token) / 20.0, 1.0)

        return [
            is_digit,
            is_alpha,
            is_alnum,
            is_upper,
            is_title,
            has_digit,
            is_decimal,
            is_large_int,
            has_comma,
            has_dot,
            is_money,
            is_time,
            is_date,
            is_single,
            norm_len,
        ]

    def extract_regex_features(self, tokens: List[str]) -> np.ndarray:
        """
        Extracts explicit features for a list of tokens.

        Args:
            tokens (List[str]): List of raw text tokens.

        Returns:
            np.ndarray: Feature matrix of shape (num_tokens, NUM_REGEX_FEATURES).
        """
        # List comprehension is generally faster than loops for this scale
        features = [self._extract_single_token_features(t) for t in tokens]
        return np.array(features, dtype=np.float32)

    def build_or_load_priors(
        self, class_vocab: Vocab, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Computes or loads the global prior probabilities for each token.
        Calculates P(class | token) based on the training set.

        Args:
            class_vocab (Vocab): Vocabulary object for classes to ensure column alignment.
            load_cached_data (bool): Whether to try loading from cache first.

        Returns:
            pd.DataFrame: Index is 'before' token, columns are class names, values are probabilities.
        """
        cache_path = Config.PRIORS_FILE

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading priors from {cache_path}...")
            try:
                priors = pd.read_parquet(cache_path)
                return priors
            except Exception as e:
                logger.warning(f"Failed to load priors from cache: {e}. Recomputing...")

        # 2. Compute from scratch
        logger.info("Computing priors from training data...")

        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load training data
        df = pd.read_csv(Config.TRAIN_FILE, dtype=str, keep_default_na=False)

        # Group by token and class to get counts
        counts = df.groupby(["before", "class"]).size().reset_index(name="count")

        # Pivot to wide format: index=token, columns=class, values=count
        priors = counts.pivot(index="before", columns="class", values="count").fillna(0)

        # Normalize rows to sum to 1 (probabilities)
        priors = priors.div(priors.sum(axis=1), axis=0)

        # Ensure all vocabulary classes are present as columns
        # This guarantees the vector size matches the class vocabulary size
        vocab_classes = [class_vocab.lookup_token(i) for i in range(len(class_vocab))]

        # Reindex columns to match vocab order, filling missing classes with 0.0
        priors = priors.reindex(columns=vocab_classes, fill_value=0.0)

        # Save to cache
        priors.to_parquet(cache_path)
        logger.info(f"Priors computed and saved to {cache_path}. Shape: {priors.shape}")

        return priors

    def build_or_load_knowledge_base(
        self, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Constructs the deterministic Knowledge Base: (token, class) -> normalized_text.
        Used for the primary retrieval path in the hybrid model.

        Args:
            load_cached_data (bool): Whether to try loading from cache first.

        Returns:
            pd.DataFrame: DataFrame with columns ['before', 'class', 'after'].
        """
        cache_path = Config.KNOWLEDGE_BASE_FILE

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading Knowledge Base from {cache_path}...")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load KB from cache: {e}. Rebuilding...")

        # 2. Build from scratch
        logger.info("Building Knowledge Base from training data...")

        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        df = pd.read_csv(Config.TRAIN_FILE, dtype=str, keep_default_na=False)

        # We want the most frequent 'after' for each (before, class) pair.
        # Group by [before, class, after] and count occurrences
        counts = (
            df.groupby(["before", "class", "after"]).size().reset_index(name="count")
        )

        # Sort by count descending so the first occurrence is the most frequent
        counts = counts.sort_values(
            ["before", "class", "count"], ascending=[True, True, False]
        )

        # Drop duplicates keeping the first (most frequent)
        kb = counts.drop_duplicates(subset=["before", "class"], keep="first")

        # Keep only relevant columns
        kb = kb[["before", "class", "after"]]

        # Save to cache
        kb.to_parquet(cache_path, index=False)
        logger.info(
            f"Knowledge Base built and saved to {cache_path}. Entries: {len(kb)}"
        )

        return kb

    def get_priors_vector(
        self, tokens: List[str], priors_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Retrieves the prior probability vectors for a list of tokens.
        If a token is not in the priors_df (OOV), returns a zero vector.

        Args:
            tokens: List of input tokens.
            priors_df: DataFrame computed by build_or_load_priors.

        Returns:
            np.ndarray: Shape (len(tokens), num_classes)
        """
        # Efficiently align tokens with the priors DataFrame
        # reindex returns a DataFrame with the same length as tokens
        # OOV tokens will have NaN rows, which we fill with 0.0
        aligned = priors_df.reindex(tokens, fill_value=0.0)

        return aligned.to_numpy(dtype=np.float32)
