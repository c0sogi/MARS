import os
import pandas as pd
import numpy as np
import nltk
from library.config import Config

# Ensure nltk resources are available if needed, though edit_distance is usually available directly.
# nltk.edit_distance is a function, no download needed for basic usage.


def get_norm_levenshtein(str1: str, str2: str) -> float:
    """
    Computes the normalized Levenshtein similarity between two strings.
    Score = 1 - (distance / max(len(str1), len(str2)))
    """
    s1 = str(str1).lower().strip()
    s2 = str(str2).lower().strip()

    if not s1 and not s2:
        return 1.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0  # Should be covered by above, but for safety

    dist = nltk.edit_distance(s1, s2)
    return 1.0 - (dist / max_len)


def get_jaccard(str1: str, str2: str) -> float:
    """
    Computes the Jaccard similarity between the sets of words in two strings.
    """
    s1 = set(str(str1).lower().split())
    s2 = set(str(str2).lower().split())

    if not s1 and not s2:
        return 1.0

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))

    return intersection / union if union > 0 else 0.0


def get_length_ratio(str1: str, str2: str) -> float:
    """
    Computes the ratio of lengths: min_len / max_len.
    """
    l1 = len(str(str1))
    l2 = len(str(str2))

    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0

    return min(l1, l2) / max(l1, l2)


def generate_structural_features(
    df: pd.DataFrame, split_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Generates deterministic structural features for a given DataFrame.

    Args:
        df: Input DataFrame containing 'anchor' and 'target' columns.
        split_name: Name of the split (e.g., 'train', 'val', 'test') for caching purposes.
        load_cached_data: Whether to try loading from cache first.

    Returns:
        pd.DataFrame: DataFrame containing the computed features.
    """
    # Define cache path
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_structural_features.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached structural features from {cache_path}")
            try:
                features_df = pd.read_parquet(cache_path)
                # Verify length matches
                if len(features_df) == len(df):
                    return features_df
                else:
                    print(
                        f"Cache length mismatch (Cache: {len(features_df)}, Input: {len(df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache file not found at {cache_path}. Computing features...")
    else:
        print("Skipping cache load. Computing features...")

    # 2. Compute features
    print(f"Generating structural features for {split_name}...")

    # Pre-allocate dictionary for speed
    features = {"norm_levenshtein": [], "jaccard": [], "len_ratio": []}

    # Iterate and compute
    # Using list comprehension or map is generally faster than df.apply
    anchors = df["anchor"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()

    for a, t in zip(anchors, targets):
        features["norm_levenshtein"].append(get_norm_levenshtein(a, t))
        features["jaccard"].append(get_jaccard(a, t))
        features["len_ratio"].append(get_length_ratio(a, t))

    features_df = pd.DataFrame(features)

    # Ensure index matches input df
    features_df.index = df.index

    # 3. Save to cache
    try:
        features_df.to_parquet(cache_path)
        print(f"Saved structural features to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return features_df
