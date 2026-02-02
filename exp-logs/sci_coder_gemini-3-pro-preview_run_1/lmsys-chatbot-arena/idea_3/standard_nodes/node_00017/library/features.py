import os
import hashlib
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import ensure_directories


class FeatureEngineer:
    """
    Handles the generation, processing, and caching of auxiliary scalar features
    for the chatbot response comparison task.
    """

    def __init__(self):
        self.epsilon = 1e-8
        # Ensure the working/cache directories exist
        ensure_directories()

    def _compute_primitives(self, df: pd.DataFrame):
        """
        Computes basic statistics (lengths, counts) for response columns.
        """
        # Ensure string type and handle NaNs
        resp_a = df["response_a"].fillna("").astype(str)
        resp_b = df["response_b"].fillna("").astype(str)

        primitives = {}

        # Character Lengths
        primitives["char_len_a"] = resp_a.str.len()
        primitives["char_len_b"] = resp_b.str.len()

        # Word Counts (simple whitespace split)
        primitives["word_len_a"] = resp_a.apply(lambda x: len(x.split()))
        primitives["word_len_b"] = resp_b.apply(lambda x: len(x.split()))

        # Newline Counts
        primitives["newline_a"] = resp_a.apply(lambda x: x.count("\n"))
        primitives["newline_b"] = resp_b.apply(lambda x: x.count("\n"))

        # Formatting Counts (Code blocks and Bold text)
        primitives["code_a"] = resp_a.apply(lambda x: x.count("```"))
        primitives["code_b"] = resp_b.apply(lambda x: x.count("```"))
        primitives["bold_a"] = resp_a.apply(lambda x: x.count("**"))
        primitives["bold_b"] = resp_b.apply(lambda x: x.count("**"))

        return primitives

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates the feature matrix based on Config.SCALAR_FEATURE_LIST.

        Args:
            df (pd.DataFrame): Input dataframe containing 'response_a' and 'response_b'.

        Returns:
            np.ndarray: A matrix of shape (N, NUM_SCALAR_FEATURES).
        """
        # Compute raw stats
        p = self._compute_primitives(df)

        feature_dict = {}

        # 1. Character Length Difference
        feature_dict["char_len_diff"] = p["char_len_a"] - p["char_len_b"]

        # 2. Word Length Difference
        feature_dict["word_len_diff"] = p["word_len_a"] - p["word_len_b"]

        # 3. Character Length Ratio
        feature_dict["char_len_ratio"] = p["char_len_a"] / (
            p["char_len_b"] + self.epsilon
        )

        # 4. Word Length Ratio
        feature_dict["word_len_ratio"] = p["word_len_a"] / (
            p["word_len_b"] + self.epsilon
        )

        # 5. Newline Difference
        feature_dict["newline_diff"] = p["newline_a"] - p["newline_b"]

        # 6. Code Block Difference
        feature_dict["code_diff"] = p["code_a"] - p["code_b"]

        # 7. Bold Text Difference
        feature_dict["bold_diff"] = p["bold_a"] - p["bold_b"]

        # Select and order features based on Config
        selected_features = []
        for feature_name in Config.SCALAR_FEATURE_LIST:
            if feature_name in feature_dict:
                selected_features.append(feature_dict[feature_name].values)
            else:
                raise ValueError(
                    f"Feature '{feature_name}' is not implemented in FeatureEngineer."
                )

        # Stack into (N, F) matrix
        feature_matrix = np.column_stack(selected_features).astype(np.float32)

        return feature_matrix

    def _get_cache_path(self, split_name: str) -> str:
        """
        Generates a cache filename based on the split name and a hash of the feature configuration.
        """
        # Create a hash of the feature list to ensure cache validity if config changes
        config_str = str(Config.SCALAR_FEATURE_LIST)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

        filename = f"{split_name}_features_{config_hash}.npy"
        return os.path.join(Config.CACHE_DIR, filename)

    def process_and_cache(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Computes features for the given dataframe, with caching.

        Args:
            df (pd.DataFrame): The dataframe containing text data.
            split_name (str): Identifier for the split (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: The computed or loaded feature matrix.
        """
        cache_path = self._get_cache_path(split_name)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(
                    f"Loading cached features for '{split_name}' from {cache_path}..."
                )
                features = np.load(cache_path)
                # Verify shape matches current config
                if features.shape[1] != len(Config.SCALAR_FEATURE_LIST):
                    print(
                        f"Cached feature dimension {features.shape[1]} mismatch with config {len(Config.SCALAR_FEATURE_LIST)}. Recomputing..."
                    )
                else:
                    return features
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Computing features for '{split_name}'...")
        features = self.extract_features(df)

        # 3. Save to cache
        try:
            np.save(cache_path, features)
            print(f"Features saved to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save features to cache: {e}")

        return features
