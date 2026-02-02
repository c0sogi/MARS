import os
import pandas as pd
import numpy as np
import nltk
from library.utils import seed_everything


class StructuralFeatureEngineer:
    """
    Computes explicit structural features for phrase pairs to augment the semantic model.
    Features:
    1. Normalized Levenshtein Distance: Character-level edit distance normalized by max length.
    2. Jaccard Similarity: Word-level intersection over union.
    3. Length Ratio: Ratio of the shorter phrase length to the longer phrase length.
    """

    def __init__(self, cache_dir="./working/idea_5"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        seed_everything(42)

    def _normalized_levenshtein(self, s1: str, s2: str) -> float:
        """
        Computes Levenshtein distance normalized by the maximum length of the two strings.
        Returns 0.0 for identical strings.
        """
        if len(s1) == 0 and len(s2) == 0:
            return 0.0

        # nltk.edit_distance computes the standard Levenshtein distance
        dist = nltk.edit_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        if max_len == 0:
            return 0.0

        return dist / max_len

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """
        Computes Jaccard similarity between sets of words (split by whitespace).
        Returns 1.0 for identical sets, 0.0 for disjoint sets.
        """
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())

        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty implies equality in this context

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def _length_ratio(self, s1: str, s2: str) -> float:
        """
        Computes the ratio of lengths (min_len / max_len).
        Returns a value between 0.0 and 1.0.
        """
        l1 = len(s1)
        l2 = len(s2)

        if l1 == 0 and l2 == 0:
            return 1.0
        if l1 == 0 or l2 == 0:
            return 0.0

        return min(l1, l2) / max(l1, l2)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates features for a given DataFrame containing 'anchor' and 'target' columns.
        """
        # Ensure inputs are strings and handle NaNs
        anchors = df["anchor"].fillna("").astype(str).tolist()
        targets = df["target"].fillna("").astype(str).tolist()

        lev_scores = []
        jaccard_scores = []
        len_ratios = []

        # Iterate and compute
        for a, t in zip(anchors, targets):
            lev_scores.append(self._normalized_levenshtein(a, t))
            jaccard_scores.append(self._jaccard_similarity(a, t))
            len_ratios.append(self._length_ratio(a, t))

        # Return as DataFrame with float32 to save memory
        return pd.DataFrame(
            {
                "norm_levenshtein": np.array(lev_scores, dtype=np.float32),
                "jaccard_sim": np.array(jaccard_scores, dtype=np.float32),
                "len_ratio": np.array(len_ratios, dtype=np.float32),
            }
        )

    def get_features(
        self, df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Orchestrates caching and computation for a specific dataset split.
        Strictly follows the caching logic: Try load -> Compute if fail -> Save.
        """
        cache_path = os.path.join(
            self.cache_dir, f"{dataset_name}_structural_features.parquet"
        )

        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                # If loading fails (corrupt), proceed to compute
                pass

        # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
        features = self.compute_features(df)

        # Save the result to the cache directory
        features.to_parquet(cache_path)

        return features


def get_all_structural_features(load_cached_data: bool = True):
    """
    Loads metadata for train, val, and test splits, and computes/loads their structural features.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_features_df, val_features_df, test_features_df)
    """
    # Load Metadata from the pre-generated files
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    engineer = StructuralFeatureEngineer()

    # Process each split
    train_feats = engineer.get_features(
        train_df, "train", load_cached_data=load_cached_data
    )
    val_feats = engineer.get_features(val_df, "val", load_cached_data=load_cached_data)
    test_feats = engineer.get_features(
        test_df, "test", load_cached_data=load_cached_data
    )

    return train_feats, val_feats, test_feats
