import os
import pandas as pd
import numpy as np
import nltk
from library.config import Config


def get_levenshtein_metrics(s1, s2):
    """
    Computes Levenshtein distance and a normalized similarity score.
    """
    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()

    if s1 == s2:
        return 0.0, 1.0

    dist = nltk.edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))

    if max_len == 0:
        # Both strings are empty but not caught by s1==s2 check (unlikely but safe)
        return 0.0, 1.0

    norm_score = 1.0 - (dist / max_len)
    return float(dist), float(norm_score)


def get_jaccard_similarity(s1, s2):
    """
    Computes Jaccard similarity between sets of words.
    """
    s1_tokens = set(str(s1).lower().split())
    s2_tokens = set(str(s2).lower().split())

    if not s1_tokens and not s2_tokens:
        return 1.0

    intersection = len(s1_tokens.intersection(s2_tokens))
    union = len(s1_tokens.union(s2_tokens))

    if union == 0:
        return 0.0

    return float(intersection) / union


def compute_features_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes structural features for a given DataFrame containing 'anchor' and 'target'.
    """
    # Pre-allocate lists for speed
    lev_dists = []
    lev_sims = []
    jaccards = []
    len_diffs = []
    len_ratios = []
    word_diffs = []

    for _, row in df.iterrows():
        a = row["anchor"]
        t = row["target"]

        # Levenshtein
        dist, sim = get_levenshtein_metrics(a, t)
        lev_dists.append(dist)
        lev_sims.append(sim)

        # Jaccard
        jac = get_jaccard_similarity(a, t)
        jaccards.append(jac)

        # Length Features
        len_a = len(str(a))
        len_t = len(str(t))

        l_diff = abs(len_a - len_t)
        # Avoid division by zero
        denom = max(len_a, len_t)
        l_ratio = min(len_a, len_t) / denom if denom > 0 else 1.0

        len_diffs.append(l_diff)
        len_ratios.append(l_ratio)

        # Word Count Diff
        wc_a = len(str(a).split())
        wc_t = len(str(t).split())
        word_diffs.append(abs(wc_a - wc_t))

    # Create result dataframe
    features_df = pd.DataFrame(
        {
            "id": df["id"].values,
            "levenshtein_dist": lev_dists,
            "levenshtein_norm": lev_sims,
            "jaccard_sim": jaccards,
            "len_diff": len_diffs,
            "len_ratio": len_ratios,
            "word_len_diff": word_diffs,
        }
    )

    return features_df


def generate_structural_features(
    df: pd.DataFrame, cache_name: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main entry point to get structural features. Handles caching logic.

    Args:
        df: Input DataFrame containing 'id', 'anchor', 'target'.
        cache_name: Identifier for the cache file (e.g., 'train', 'val', 'test').
        load_cached_data: Whether to try loading from disk.

    Returns:
        DataFrame containing 'id' and the computed feature columns.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(
        Config.WORKING_DIR, f"{cache_name}_structural_features.parquet"
    )

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached structural features from {cache_path}")
            cached_df = pd.read_parquet(cache_path)
            # Verify IDs match to ensure cache validity
            if len(cached_df) == len(df) and (cached_df["id"] == df["id"]).all():
                return cached_df
            else:
                # print("Cache ID mismatch. Recomputing...")
                pass
        except Exception as e:
            # print(f"Error loading cache: {e}. Recomputing...")
            pass

    # 2. Compute from scratch
    # print(f"Computing structural features for {cache_name}...")
    features_df = compute_features_raw(df)

    # 3. Save to cache
    try:
        features_df.to_parquet(cache_path, index=False)
        # print(f"Saved structural features to {cache_path}")
    except Exception as e:
        # print(f"Failed to save cache: {e}")
        pass

    return features_df
