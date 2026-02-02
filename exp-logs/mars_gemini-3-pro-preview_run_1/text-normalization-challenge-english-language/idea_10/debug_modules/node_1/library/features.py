import re
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import load_or_create_cache


class ExplicitFeatureExtractor:
    """
    Extracts explicit morphological features from text tokens using Regex.
    Generates a binary vector of size Config.NUM_REGEX_FEATURES (16).
    """

    def __init__(self):
        self.num_features = Config().NUM_REGEX_FEATURES

        # Define patterns corresponding to the 16 features.
        # These patterns cover morphological cues relevant to text normalization.
        self.patterns = [
            # 1. Is Digit (Integers) - e.g., "123"
            r"^\d+$",
            # 2. Is Decimal (Float) - e.g., "3.14"
            r"^\d*\.\d+$",
            # 3. Is Ordinal - e.g., "1st", "2nd"
            r"^\d+(st|nd|rd|th)$",
            # 4. Is Roman Numeral - e.g., "IV", "MCML"
            r"^[IVXLCDM]+$",
            # 5. Is Alpha (Word) - e.g., "hello"
            r"^[a-zA-Z]+$",
            # 6. Is Upper (Acronyms/Caps) - e.g., "USA"
            r"^[A-Z]+$",
            # 7. Is Title Case - e.g., "Monday"
            r"^[A-Z][a-z]+",
            # 8. Is Punctuation - e.g., ",", "."
            r"^[^\w\s]+$",
            # 9. Has Currency Symbol - e.g., "$", "€"
            r"[$£€¥¢]",
            # 10. Is Time Format - e.g., "10:30"
            r"\d{1,2}:\d{2}",
            # 11. Is Date Format (Slash/Dash) - e.g., "2023-01-01", "12/05"
            r"\d{1,4}[-/]\d{1,4}",
            # 12. Is Measure Unit (Common multi-char units) - e.g., "kg", "MHz"
            r"\b(kg|mm|cm|km|mg|ml|oz|lb|ft|Hz|MHz|GHz)\b",
            # 13. Is URL or Email - e.g., "http", ".com", "@"
            r"(http|www|\.com|@)",
            # 14. Has Digit (Anywhere) - e.g., "Model-3"
            r"\d",
            # 15. Has Dash or Slash (Separators) - e.g., "one-way"
            r"[-/]",
            # 16. Is Alphanumeric (Mixed) - e.g., "A100"
            r"^[a-zA-Z0-9]+$",
        ]

        # Validation to ensure alignment with Config
        if len(self.patterns) != self.num_features:
            raise ValueError(
                f"ExplicitFeatureExtractor defined {len(self.patterns)} patterns, "
                f"but Config expects {self.num_features}."
            )

    def transform_series(self, series: pd.Series) -> np.ndarray:
        """
        Optimized transformation for a Pandas Series (Bulk Processing).

        Args:
            series (pd.Series): Series of text tokens.

        Returns:
            np.ndarray: Feature matrix of shape (len(series), num_features)
        """
        # Ensure input is string type to avoid errors with numbers/NaNs
        series = series.astype(str)

        feature_cols = []
        for pat in self.patterns:
            # vectorized regex matching
            # astype(np.float32) converts boolean True/False to 1.0/0.0
            mask = series.str.contains(pat, regex=True).astype(np.float32)
            feature_cols.append(mask)

        # Stack columns efficiently
        features_df = pd.concat(feature_cols, axis=1)
        return features_df.values

    def transform_single(self, token: str) -> np.ndarray:
        """
        Transform a single token (Inference/Loop Processing).

        Args:
            token (str): Input token.

        Returns:
            np.ndarray: Feature vector of shape (num_features,)
        """
        token = str(token)
        feats = []
        for pat in self.patterns:
            if re.search(pat, token):
                feats.append(1.0)
            else:
                feats.append(0.0)
        return np.array(feats, dtype=np.float32)


def _compute_features_wrapper(df):
    """
    Internal wrapper function to be passed to load_or_create_cache.
    Extracts features from the 'before' column of the dataframe.
    """
    extractor = ExplicitFeatureExtractor()
    print(f"Extracting explicit features for {len(df)} tokens...")
    return extractor.transform_series(df["before"])


def get_cached_features(df, cache_name, load_cached_data=True):
    """
    Retrieves explicit features for a dataframe, utilizing the caching mechanism.

    Args:
        df (pd.DataFrame): Dataframe containing the 'before' column.
        cache_name (str): Unique identifier for the cache file (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        np.ndarray: Feature matrix of shape (N, 16).
    """
    config = Config()
    cache_path = os.path.join(
        config.WORKING_DIR, "cache", f"{cache_name}_explicit_features.npy"
    )

    return load_or_create_cache(
        file_path=cache_path,
        compute_func=_compute_features_wrapper,
        load_cached_data=load_cached_data,
        df=df,
    )
