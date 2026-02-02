import os
import pandas as pd
import numpy as np
from library.config import WORKING_DIR, ModelConfig

# =============================================================================
# CONSTANTS & PATHS
# =============================================================================

CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")

# Parquet filenames
FN_TRIGRAM = "trigram.parquet"
FN_BIGRAM_PREV = "bigram_prev.parquet"
FN_BIGRAM_NEXT = "bigram_next.parquet"
FN_UNIGRAM = "unigram.parquet"

# =============================================================================
# HFBB MODEL CLASS
# =============================================================================


class HFBBModel:
    """
    Tier 1: Confidence-Aware Granular Memory (Hierarchical Frequency-Based Backoff).

    Implements a 4-level backoff strategy:
    1. Trigram (Prev, Curr, Next)
    2. Bigram Prev (Prev, Curr)
    3. Bigram Next (Curr, Next)
    4. Unigram (Curr) -> Gated by Confidence Score
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # In-memory lookup tables (Dicts for O(1) access)
        # Keys are tuples of strings, values are strings (or tuples for unigram)
        self.trigram_dict = {}
        self.bigram_prev_dict = {}
        self.bigram_next_dict = {}
        self.unigram_dict = {}  # Value: (prediction, confidence)

    def build(self, df_train: pd.DataFrame, load_cached_data=True):
        """
        Builds the statistical lookup tables from the training data.

        Args:
            df_train: DataFrame containing 'sentence_id', 'token_id', 'before', 'after'.
            load_cached_data: If True, attempts to load from Parquet cache.
        """
        # Paths
        path_tri = os.path.join(CACHE_DIR, FN_TRIGRAM)
        path_bi_p = os.path.join(CACHE_DIR, FN_BIGRAM_PREV)
        path_bi_n = os.path.join(CACHE_DIR, FN_BIGRAM_NEXT)
        path_uni = os.path.join(CACHE_DIR, FN_UNIGRAM)

        # Check if cache exists
        cache_exists = (
            os.path.exists(path_tri)
            and os.path.exists(path_bi_p)
            and os.path.exists(path_bi_n)
            and os.path.exists(path_uni)
        )

        if load_cached_data and cache_exists:
            print("Loading HFBB lookup tables from cache...")
            df_tri = pd.read_parquet(path_tri)
            df_bi_p = pd.read_parquet(path_bi_p)
            df_bi_n = pd.read_parquet(path_bi_n)
            df_uni = pd.read_parquet(path_uni)
        else:
            print("Computing HFBB statistics from source data...")
            # 1. Prepare Context
            # We need to shift columns to get prev/next, respecting sentence boundaries
            df_proc = df_train.copy()
            df_proc["before"] = df_proc["before"].fillna("").astype(str)
            df_proc["after"] = df_proc["after"].fillna("").astype(str)

            # Shift for Prev
            df_proc["prev"] = df_proc["before"].shift(1).fillna("")
            # Mask where sentence changed
            mask_prev = df_proc["sentence_id"] != df_proc["sentence_id"].shift(1)
            df_proc.loc[mask_prev, "prev"] = ""

            # Shift for Next
            df_proc["next"] = df_proc["before"].shift(-1).fillna("")
            # Mask where sentence changed
            mask_next = df_proc["sentence_id"] != df_proc["sentence_id"].shift(-1)
            df_proc.loc[mask_next, "next"] = ""

            # 2. Compute Aggregations
            # Helper to get mode efficiently: Group -> Count -> Sort -> Drop Duplicates
            def get_mode_table(df, group_cols):
                # Count occurrences of (Context + Target)
                counts = (
                    df.groupby(group_cols + ["after"]).size().reset_index(name="count")
                )
                # Sort by count desc
                counts = counts.sort_values("count", ascending=False)
                # Drop duplicates on context to keep only the mode
                modes = counts.drop_duplicates(subset=group_cols).drop(
                    columns=["count"]
                )
                return modes

            print("  - Computing Trigrams...")
            df_tri = get_mode_table(df_proc, ["prev", "before", "next"])

            print("  - Computing Bigrams (Prev)...")
            df_bi_p = get_mode_table(df_proc, ["prev", "before"])

            print("  - Computing Bigrams (Next)...")
            df_bi_n = get_mode_table(df_proc, ["before", "next"])

            print("  - Computing Unigrams with Confidence...")
            # Unigram needs confidence score
            # 1. Get counts of (before, after)
            uni_counts = (
                df_proc.groupby(["before", "after"]).size().reset_index(name="count")
            )
            # 2. Get total counts of (before)
            uni_totals = df_proc.groupby("before").size().reset_index(name="total")
            # 3. Find mode
            uni_counts = uni_counts.sort_values("count", ascending=False)
            df_uni = uni_counts.drop_duplicates(subset=["before"])
            # 4. Merge total to calculate confidence
            df_uni = df_uni.merge(uni_totals, on="before", how="left")
            df_uni["confidence"] = df_uni["count"] / df_uni["total"]
            # Keep only relevant columns
            df_uni = df_uni[["before", "after", "confidence"]]

            # 3. Save to Cache
            print("Saving HFBB tables to cache...")
            df_tri.to_parquet(path_tri, index=False)
            df_bi_p.to_parquet(path_bi_p, index=False)
            df_bi_n.to_parquet(path_bi_n, index=False)
            df_uni.to_parquet(path_uni, index=False)

        # 4. Populate In-Memory Dictionaries
        print("Populating in-memory dictionaries...")

        # Trigram: (prev, curr, next) -> after
        # Using zip is faster than iterrows
        self.trigram_dict = {
            (p, c, n): a
            for p, c, n, a in zip(
                df_tri["prev"], df_tri["before"], df_tri["next"], df_tri["after"]
            )
        }

        # Bigram Prev: (prev, curr) -> after
        self.bigram_prev_dict = {
            (p, c): a
            for p, c, a in zip(df_bi_p["prev"], df_bi_p["before"], df_bi_p["after"])
        }

        # Bigram Next: (curr, next) -> after
        self.bigram_next_dict = {
            (c, n): a
            for c, n, a in zip(df_bi_n["before"], df_bi_n["next"], df_bi_n["after"])
        }

        # Unigram: curr -> (after, confidence)
        self.unigram_dict = {
            c: (a, conf)
            for c, a, conf in zip(
                df_uni["before"], df_uni["after"], df_uni["confidence"]
            )
        }

        print(
            f"HFBB Model Ready. "
            f"Tri: {len(self.trigram_dict)}, "
            f"BiP: {len(self.bigram_prev_dict)}, "
            f"BiN: {len(self.bigram_next_dict)}, "
            f"Uni: {len(self.unigram_dict)}"
        )

    def query(self, token, prev_token="", next_token=""):
        """
        Queries the hierarchical model for a normalization.

        Args:
            token: The current token string.
            prev_token: The previous token string (context).
            next_token: The next token string (context).

        Returns:
            str: The predicted normalization string.
            None: If no confident match is found (signal to use Tier 2).
        """
        # Ensure inputs are strings (handle potential None/NaN passed from outside)
        t = str(token) if token is not None else ""
        p = str(prev_token) if prev_token is not None else ""
        n = str(next_token) if next_token is not None else ""

        # 1. Trigram Check
        res = self.trigram_dict.get((p, t, n))
        if res is not None:
            return res

        # 2. Bigram Prev Check
        res = self.bigram_prev_dict.get((p, t))
        if res is not None:
            return res

        # 3. Bigram Next Check
        res = self.bigram_next_dict.get((t, n))
        if res is not None:
            return res

        # 4. Unigram Check (with Confidence Gating)
        uni_res = self.unigram_dict.get(t)
        if uni_res is not None:
            prediction, confidence = uni_res
            if confidence >= self.confidence_threshold:
                return prediction

        # Fallback to Tier 2 (Transformer)
        return None
