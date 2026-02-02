import os
import pandas as pd
import numpy as np
import nltk
from library.config import CFG


def get_normalized_levenshtein(s1, s2):
    """
    Calculates the normalized Levenshtein distance between two strings.
    Returns a value between 0.0 (identical) and 1.0 (completely different).
    """
    s1 = str(s1)
    s2 = str(s2)
    len1 = len(s1)
    len2 = len(s2)
    max_len = max(len1, len2)

    if max_len == 0:
        return 0.0

    # nltk.edit_distance calculates the minimum number of operations
    dist = nltk.edit_distance(s1, s2)
    return dist / max_len


def get_jaccard_similarity(s1, s2):
    """
    Calculates the Jaccard similarity between the sets of words in two strings.
    Returns a value between 0.0 (no overlap) and 1.0 (identical set of words).
    """
    # Simple whitespace tokenization and lowercasing
    tokens1 = set(str(s1).lower().split())
    tokens2 = set(str(s2).lower().split())

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    if union == 0:
        return 0.0

    return intersection / union


def get_length_ratio(s1, s2):
    """
    Calculates the length ratio between two strings.
    Uses min/max to ensure the result is in [0, 1] and symmetric.
    """
    s1 = str(s1)
    s2 = str(s2)
    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    return min(len1, len2) / max(len1, len2)


def get_features_batch(anchors, targets, cache_name="features", load_cached_data=True):
    """
    Generates a dense vector of structural features for batches of anchor and target pairs.
    Implements caching using parquet files in the working directory.

    Args:
        anchors (iterable): Iterable of anchor strings.
        targets (iterable): Iterable of target strings.
        cache_name (str): Unique identifier for the cache file (e.g., 'train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: A float32 numpy array of shape (N, 3) containing:
                    [normalized_levenshtein, jaccard_similarity, length_ratio]
    """
    # Ensure the working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    cache_path = os.path.join(
        CFG.working_dir, f"{cache_name}_structural_features.parquet"
    )

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure columns are in the expected order
            expected_cols = ["levenshtein", "jaccard", "len_ratio"]
            if all(col in df.columns for col in expected_cols):
                return df[expected_cols].values.astype(np.float32)
        except Exception:
            # If loading fails (e.g. corrupt file), proceed to recompute
            pass

    # 2. Compute features from scratch
    features_list = []

    # Convert to lists to ensure safe iteration (e.g. if inputs are generators or Series with different indices)
    anchors_list = list(anchors)
    targets_list = list(targets)

    for anchor, target in zip(anchors_list, targets_list):
        lev = get_normalized_levenshtein(anchor, target)
        jac = get_jaccard_similarity(anchor, target)
        lr = get_length_ratio(anchor, target)
        features_list.append([lev, jac, lr])

    df = pd.DataFrame(features_list, columns=["levenshtein", "jaccard", "len_ratio"])

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df.values.astype(np.float32)
