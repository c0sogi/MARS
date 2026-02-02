import os
import pandas as pd
import numpy as np
from library.config import Config


class HFBBModel:
    """
    Hierarchical Frequency Back-off (HFBB) Model.

    This model acts as Tier 1 in the cascade. It stores statistical mappings
    from input contexts to normalized text.

    Hierarchy:
    1. Trigram: (prev, current, next) -> target
    2. Bigram Prev: (prev, current) -> target
    3. Bigram Next: (current, next) -> target
    4. Unigram: current -> (target, confidence)

    The Unigram layer includes a confidence score: P(mode) = count(mode) / total_count.
    """

    def __init__(self):
        # Maps for O(1) lookup
        self.unigram_map = {}  # key: current -> value: (target, confidence)
        self.bigram_prev_map = {}  # key: (prev, current) -> value: target
        self.bigram_next_map = {}  # key: (current, next) -> value: target
        self.trigram_map = {}  # key: (prev, current, next) -> value: target

        # Cache filenames
        self.cache_files = {
            "unigram": os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet"),
            "bigram_prev": os.path.join(Config.HFBB_CACHE_DIR, "bigram_prev.parquet"),
            "bigram_next": os.path.join(Config.HFBB_CACHE_DIR, "bigram_next.parquet"),
            "trigram": os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet"),
        }

    def fit(self, df: pd.DataFrame, load_cached_data: bool = True) -> None:
        """
        Computes n-gram statistics from the training dataframe.
        Uses caching to avoid re-computation.

        Args:
            df (pd.DataFrame): Training data containing 'sentence_id', 'before', 'after'.
            load_cached_data (bool): Whether to attempt loading from disk.
        """
        # Ensure cache directory exists
        os.makedirs(Config.HFBB_CACHE_DIR, exist_ok=True)

        all_cached = all(os.path.exists(p) for p in self.cache_files.values())

        if load_cached_data and all_cached:
            print("Loading HFBB stats from cache...")
            self._load_cache()
        else:
            print("Computing HFBB stats from scratch...")
            self._compute_stats(df)

    def _compute_stats(self, df: pd.DataFrame):
        """
        Internal method to compute statistics and save to cache.
        """
        # 1. Preprocess Contexts
        # We need efficient shifting respecting sentence boundaries
        # Assuming df is sorted by sentence_id, token_id (standard for this dataset)

        # Create working copy with string types to ensure consistency
        work_df = df[["sentence_id", "before", "after"]].copy()
        work_df["before"] = work_df["before"].astype(str)
        work_df["after"] = work_df["after"].astype(str)

        # Shift for context
        # Using a mask for sentence boundaries is faster than groupby().shift()
        # prev token
        work_df["prev"] = work_df["before"].shift(1)
        work_df.loc[
            work_df["sentence_id"] != work_df["sentence_id"].shift(1), "prev"
        ] = "<start>"
        work_df["prev"] = work_df["prev"].fillna("<start>")

        # next token
        work_df["next"] = work_df["before"].shift(-1)
        work_df.loc[
            work_df["sentence_id"] != work_df["sentence_id"].shift(-1), "next"
        ] = "<end>"
        work_df["next"] = work_df["next"].fillna("<end>")

        # ---------------------------------------------------------
        # 2. Compute Unigram Stats (with Confidence)
        # ---------------------------------------------------------
        print("Computing Unigrams...")
        # Count occurrences of (before, after)
        uni_counts = (
            work_df.groupby(["before", "after"]).size().reset_index(name="count")
        )

        # Calculate total counts per 'before' token
        uni_totals = uni_counts.groupby("before")["count"].transform("sum")
        uni_counts["confidence"] = uni_counts["count"] / uni_totals

        # Keep only the most frequent 'after' for each 'before'
        # Sort by count desc, then drop duplicates keeping first
        uni_best = uni_counts.sort_values(["before", "count"], ascending=[True, False])
        uni_best = uni_best.drop_duplicates(subset=["before"], keep="first")

        # Save and Populate
        uni_best.to_parquet(self.cache_files["unigram"], index=False)
        self.unigram_map = {
            row.before: (row.after, row.confidence)
            for row in uni_best.itertuples(index=False)
        }

        # ---------------------------------------------------------
        # 3. Compute Bigram Prev Stats
        # ---------------------------------------------------------
        print("Computing Bigrams (Prev)...")
        bi_prev_counts = (
            work_df.groupby(["prev", "before", "after"])
            .size()
            .reset_index(name="count")
        )
        bi_prev_best = bi_prev_counts.sort_values(
            ["prev", "before", "count"], ascending=[True, True, False]
        )
        bi_prev_best = bi_prev_best.drop_duplicates(
            subset=["prev", "before"], keep="first"
        )

        bi_prev_best.to_parquet(self.cache_files["bigram_prev"], index=False)
        self.bigram_prev_map = {
            (row.prev, row.before): row.after
            for row in bi_prev_best.itertuples(index=False)
        }

        # ---------------------------------------------------------
        # 4. Compute Bigram Next Stats
        # ---------------------------------------------------------
        print("Computing Bigrams (Next)...")
        bi_next_counts = (
            work_df.groupby(["before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        bi_next_best = bi_next_counts.sort_values(
            ["before", "next", "count"], ascending=[True, True, False]
        )
        bi_next_best = bi_next_best.drop_duplicates(
            subset=["before", "next"], keep="first"
        )

        bi_next_best.to_parquet(self.cache_files["bigram_next"], index=False)
        self.bigram_next_map = {
            (row.before, row.next): row.after
            for row in bi_next_best.itertuples(index=False)
        }

        # ---------------------------------------------------------
        # 5. Compute Trigram Stats
        # ---------------------------------------------------------
        print("Computing Trigrams...")
        tri_counts = (
            work_df.groupby(["prev", "before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        tri_best = tri_counts.sort_values(
            ["prev", "before", "next", "count"], ascending=[True, True, True, False]
        )
        tri_best = tri_best.drop_duplicates(
            subset=["prev", "before", "next"], keep="first"
        )

        tri_best.to_parquet(self.cache_files["trigram"], index=False)
        self.trigram_map = {
            (row.prev, row.before, row.next): row.after
            for row in tri_best.itertuples(index=False)
        }

        print("HFBB stats computed and cached.")

    def _load_cache(self):
        """
        Loads statistics from parquet files into memory.
        """
        # Unigram
        uni_df = pd.read_parquet(self.cache_files["unigram"])
        self.unigram_map = {
            row.before: (row.after, row.confidence)
            for row in uni_df.itertuples(index=False)
        }

        # Bigram Prev
        bi_prev_df = pd.read_parquet(self.cache_files["bigram_prev"])
        self.bigram_prev_map = {
            (row.prev, row.before): row.after
            for row in bi_prev_df.itertuples(index=False)
        }

        # Bigram Next
        bi_next_df = pd.read_parquet(self.cache_files["bigram_next"])
        self.bigram_next_map = {
            (row.before, row.next): row.after
            for row in bi_next_df.itertuples(index=False)
        }

        # Trigram
        tri_df = pd.read_parquet(self.cache_files["trigram"])
        self.trigram_map = {
            (row.prev, row.before, row.next): row.after
            for row in tri_df.itertuples(index=False)
        }

    def predict(self, token: str, prev_token: str = None, next_token: str = None):
        """
        Predicts the normalized text using the back-off hierarchy.

        Args:
            token (str): The current token to normalize.
            prev_token (str, optional): The previous token. Defaults to None (<start>).
            next_token (str, optional): The next token. Defaults to None (<end>).

        Returns:
            tuple: (prediction_str, level_name, confidence_score)
                   prediction_str is None if no match found in any layer.
        """
        token = str(token)
        prev_t = str(prev_token) if prev_token is not None else "<start>"
        next_t = str(next_token) if next_token is not None else "<end>"

        # 1. Trigram Check
        tri_key = (prev_t, token, next_t)
        if tri_key in self.trigram_map:
            return self.trigram_map[tri_key], "trigram", 1.0

        # 2. Bigram Prev Check
        bi_prev_key = (prev_t, token)
        if bi_prev_key in self.bigram_prev_map:
            return self.bigram_prev_map[bi_prev_key], "bigram_prev", 1.0

        # 3. Bigram Next Check
        bi_next_key = (token, next_t)
        if bi_next_key in self.bigram_next_map:
            return self.bigram_next_map[bi_next_key], "bigram_next", 1.0

        # 4. Unigram Check
        if token in self.unigram_map:
            pred, conf = self.unigram_map[token]
            return pred, "unigram", conf

        # 5. No Match
        return None, "none", 0.0

    def get_unigram_confidence(self, token: str) -> float:
        """
        Helper to get just the unigram confidence for a token.
        Used by the curriculum builder to identify 'Hard Positives'.
        """
        token = str(token)
        if token in self.unigram_map:
            return self.unigram_map[token][1]
        return 0.0
