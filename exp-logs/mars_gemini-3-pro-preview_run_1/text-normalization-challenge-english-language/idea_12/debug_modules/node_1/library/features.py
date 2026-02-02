import os
import re
import numpy as np
import pandas as pd
from library.config import Config


class RegexFeatureExtractor:
    """
    Extracts explicit morphological features from text tokens using
    pre-defined regex patterns.
    """

    def __init__(self):
        self.patterns = Config.REGEX_PATTERNS
        self.num_features = len(self.patterns)
        # We don't compile here if we use pandas vectorized operations,
        # but we compile for single-instance inference support if needed later.
        self.compiled_patterns = [re.compile(p) for p in self.patterns]

    def extract(self, tokens):
        """
        Extract features for a list of tokens or a pandas Series.

        Args:
            tokens: List[str] or pd.Series containing text tokens.

        Returns:
            np.ndarray: Binary feature matrix of shape (n_tokens, n_patterns).
        """
        # If input is a list, convert to Series for vectorized processing
        if isinstance(tokens, list):
            series = pd.Series(tokens, dtype=str)
        elif isinstance(tokens, pd.Series):
            series = tokens.astype(str)
        else:
            raise ValueError("Input must be a list or pandas Series")

        # Use pandas vectorized string operations for speed
        feature_list = []
        for pat in self.patterns:
            # regex=True ensures it treats the pattern as a regex
            # astype(np.uint8) converts boolean to 0/1 efficiently
            matches = series.str.contains(pat, regex=True).astype(np.uint8)
            feature_list.append(matches.values)

        # Stack to create (N, F) matrix
        # Transpose is not needed if we stack as columns, but np.stack stacks on new axis.
        # We want (N, F). feature_list is a list of N-sized arrays.
        # np.column_stack does exactly what we want.
        features = np.column_stack(feature_list)

        return features


def generate_and_cache_features(dataset_type, load_cached_data=True):
    """
    Generates regex features for the specified dataset and caches them.

    Args:
        dataset_type (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        np.ndarray: The feature matrix.
    """
    # Determine paths based on dataset type
    if dataset_type == "train":
        data_path = Config.TRAIN_DATA
        cache_path = Config.TRAIN_FEATURES_PATH
    elif dataset_type == "val":
        data_path = Config.VAL_DATA
        cache_path = Config.VAL_FEATURES_PATH
    elif dataset_type == "test":
        data_path = Config.TEST_DATA
        cache_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_type} features from {cache_path}...")
        try:
            features = np.load(cache_path)
            return features
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Generate from scratch
    print(f"Generating {dataset_type} features from {data_path}...")

    # Load raw text data
    # We only need the 'before' column
    df = pd.read_csv(data_path, usecols=["before"], dtype=str, keep_default_na=False)

    # Initialize extractor
    extractor = RegexFeatureExtractor()

    # Extract features
    # Passing the series directly allows for vectorized operations
    features = extractor.extract(df["before"])

    # 3. Save to cache
    print(f"Saving {dataset_type} features to {cache_path}...")
    np.save(cache_path, features)

    return features
