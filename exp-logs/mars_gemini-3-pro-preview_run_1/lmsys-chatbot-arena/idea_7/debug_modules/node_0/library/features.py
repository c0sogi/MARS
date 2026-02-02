import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything


class StructuralFeatureGenerator:
    """
    Generates explicit structural features for the Siamese Network.
    Focuses on relative differences and ratios between Response A and Response B.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        # Epsilon to prevent division by zero
        self.epsilon = 1e-6

    def _compute_text_stats(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Computes raw statistics for a series of text.
        """
        # Ensure string type and handle NaNs
        text_series = text_series.fillna("").astype(str)

        stats = pd.DataFrame()
        stats["char_len"] = text_series.apply(len)
        stats["word_len"] = text_series.apply(lambda x: len(x.split()))
        stats["newline_count"] = text_series.apply(lambda x: x.count("\n"))

        return stats

    def _process_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes relative structural features from the dataframe.
        Returns a numpy array of shape (N, num_features).
        """
        # Compute stats for Response A
        stats_a = self._compute_text_stats(df["response_a"])

        # Compute stats for Response B
        stats_b = self._compute_text_stats(df["response_b"])

        features = pd.DataFrame()

        # 1. Character Length Features
        features["char_len_diff"] = stats_a["char_len"] - stats_b["char_len"]
        features["char_len_ratio"] = stats_a["char_len"] / (
            stats_b["char_len"] + self.epsilon
        )

        # 2. Word Length Features
        features["word_len_diff"] = stats_a["word_len"] - stats_b["word_len"]
        features["word_len_ratio"] = stats_a["word_len"] / (
            stats_b["word_len"] + self.epsilon
        )

        # 3. Newline Count Features
        features["newline_diff"] = stats_a["newline_count"] - stats_b["newline_count"]
        features["newline_ratio"] = stats_a["newline_count"] / (
            stats_b["newline_count"] + self.epsilon
        )

        # Convert to numpy array (float32 for ML models)
        return features.values.astype(np.float32)

    def get_features(
        self, split: str = "train", load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Retrieves structural features for a specific split.
        Implements caching logic.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            np.ndarray: Feature matrix.
        """
        valid_splits = ["train", "val", "test"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

        filename = f"{split}_structural_features.npy"
        filepath = os.path.join(self.cache_dir, filename)

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(filepath):
            print(
                f"Loading cached structural features for '{split}' from {filepath}..."
            )
            try:
                features = np.load(filepath)
                return features
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Computing structural features for '{split}'...")

        # Determine source file based on split
        if split == "train":
            source_path = Config.TRAIN_PATH
        elif split == "val":
            source_path = Config.VAL_PATH
        else:
            source_path = Config.TEST_PATH

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        df = pd.read_csv(source_path)
        features = self._process_dataframe(df)

        # 3. Save to cache
        print(f"Saving structural features to {filepath}...")
        np.save(filepath, features)

        return features


def generate_all_features(load_cached_data: bool = True):
    """
    Helper function to generate features for all splits.
    """
    seed_everything()
    generator = StructuralFeatureGenerator()

    train_feats = generator.get_features("train", load_cached_data=load_cached_data)
    val_feats = generator.get_features("val", load_cached_data=load_cached_data)
    test_feats = generator.get_features("test", load_cached_data=load_cached_data)

    print("\nFeature Generation Summary:")
    print(f"Train features shape: {train_feats.shape}")
    print(f"Val features shape:   {val_feats.shape}")
    print(f"Test features shape:  {test_feats.shape}")

    return train_feats, val_feats, test_feats
