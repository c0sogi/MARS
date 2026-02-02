import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import load_metadata


class HFBBStats:
    """
    Tier 1: Confidence-Aware Granular Memory (HFBB)
    Implements a hierarchical frequency backoff model with confidence gating.

    This module acts as the high-precision filter in the Hybrid Cascade.
    It memorizes frequent patterns (Trigrams, Bigrams) and statistical modes (Unigrams).
    Unigrams carry a confidence score to determine if a token is 'stable' enough
    to be handled by memory or if it requires the neural network (Tier 2).
    """

    def __init__(self, config: Config):
        self.config = config

        # In-memory lookup dictionaries
        self.trigram_dict = {}
        self.bigram_prev_dict = {}
        self.bigram_next_dict = {}
        self.unigram_dict = {}

        # Define cache file paths using the config's artifact path generator
        # This ensures cache validity based on config hash (prevents stale data)
        self.cache_files = {
            "trigram": self.config.get_artifact_path("hfbb_trigram.parquet"),
            "bigram_prev": self.config.get_artifact_path("hfbb_bigram_prev.parquet"),
            "bigram_next": self.config.get_artifact_path("hfbb_bigram_next.parquet"),
            "unigram": self.config.get_artifact_path("hfbb_unigram.parquet"),
        }

    def fit(self, load_cached_data=True):
        """
        Computes or loads the hierarchical statistics from the training corpus.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
        """
        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in self.cache_files.values())

        if load_cached_data and cache_exists:
            print(
                f"Loading HFBB stats from cache: {os.path.dirname(self.cache_files['trigram'])}"
            )
            self._load_cache()
            return

        print("Computing HFBB stats from scratch...")

        # Load training data
        df = load_metadata("train")

        # Ensure correct ordering of tokens within sentences
        # token_id is loaded as string, convert to int for numerical sorting
        if "token_id" in df.columns:
            df["token_id_int"] = df["token_id"].astype(int)
            # Sort by sentence and then token index to reconstruct sequence
            df.sort_values(["sentence_id", "token_id_int"], inplace=True)
        else:
            print("Warning: token_id not found in metadata, assuming implicit order.")

        # --- Context Generation ---
        # Create shifted columns for context (prev/next tokens)

        # Shift to get previous and next tokens
        df["prev_token"] = df["before"].shift(1).fillna("<START>")
        df["next_token"] = df["before"].shift(-1).fillna("<END>")

        # Shift sentence_id to detect sentence boundaries
        df["prev_sent"] = df["sentence_id"].shift(1)
        df["next_sent"] = df["sentence_id"].shift(-1)

        # Apply boundaries: if sentence_id changes, context is invalid (use markers)
        # If prev row is different sentence, prev_token is <START>
        mask_start = df["prev_sent"] != df["sentence_id"]
        df.loc[mask_start, "prev_token"] = "<START>"

        # If next row is different sentence, next_token is <END>
        mask_end = df["next_sent"] != df["sentence_id"]
        df.loc[mask_end, "next_token"] = "<END>"

        # --- 1. Trigrams: P(after | prev, cur, next) ---
        print("Computing Trigrams...")
        # Count occurrences of each mapping
        tri_counts = (
            df.groupby(["prev_token", "before", "next_token", "after"])
            .size()
            .reset_index(name="count")
        )
        # Sort by count descending to put most frequent first
        tri_counts.sort_values("count", ascending=False, inplace=True)
        # Drop duplicates to keep only the mode (most frequent normalization for this context)
        tri_best = tri_counts.drop_duplicates(
            subset=["prev_token", "before", "next_token"]
        )

        # --- 2. Bigrams (Prev): P(after | prev, cur) ---
        print("Computing Bigrams (Prev)...")
        bi_prev_counts = (
            df.groupby(["prev_token", "before", "after"])
            .size()
            .reset_index(name="count")
        )
        bi_prev_counts.sort_values("count", ascending=False, inplace=True)
        bi_prev_best = bi_prev_counts.drop_duplicates(subset=["prev_token", "before"])

        # --- 3. Bigrams (Next): P(after | cur, next) ---
        print("Computing Bigrams (Next)...")
        bi_next_counts = (
            df.groupby(["before", "next_token", "after"])
            .size()
            .reset_index(name="count")
        )
        bi_next_counts.sort_values("count", ascending=False, inplace=True)
        bi_next_best = bi_next_counts.drop_duplicates(subset=["before", "next_token"])

        # --- 4. Unigrams: P(after | cur) with Confidence ---
        print("Computing Unigrams...")
        uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Calculate total occurrences of each 'before' token to compute probability
        uni_totals = uni_counts.groupby("before")["count"].transform("sum")
        uni_counts["confidence"] = uni_counts["count"] / uni_totals

        # Get mode
        uni_counts.sort_values("count", ascending=False, inplace=True)
        uni_best = uni_counts.drop_duplicates(subset=["before"])

        # --- Save and Populate ---
        print("Saving HFBB stats to cache...")
        self._save_cache(tri_best, bi_prev_best, bi_next_best, uni_best)

        # Populate in-memory dicts
        self._populate_dicts(tri_best, bi_prev_best, bi_next_best, uni_best)
        print("HFBB Stats ready.")

    def _populate_dicts(self, tri, bi_prev, bi_next, uni):
        """Converts DataFrames to optimized dictionary lookups."""
        # Trigram: (prev, cur, next) -> after
        self.trigram_dict = tri.set_index(["prev_token", "before", "next_token"])[
            "after"
        ].to_dict()

        # Bigram Prev: (prev, cur) -> after
        self.bigram_prev_dict = bi_prev.set_index(["prev_token", "before"])[
            "after"
        ].to_dict()

        # Bigram Next: (cur, next) -> after
        self.bigram_next_dict = bi_next.set_index(["before", "next_token"])[
            "after"
        ].to_dict()

        # Unigram: cur -> (after, confidence)
        # Using zip and dict constructor is faster than iterating
        self.unigram_dict = dict(
            zip(uni["before"], zip(uni["after"], uni["confidence"]))
        )

    def _save_cache(self, tri, bi_prev, bi_next, uni):
        """Saves DataFrames to Parquet."""
        tri.to_parquet(self.cache_files["trigram"], index=False)
        bi_prev.to_parquet(self.cache_files["bigram_prev"], index=False)
        bi_next.to_parquet(self.cache_files["bigram_next"], index=False)
        uni.to_parquet(self.cache_files["unigram"], index=False)

    def _load_cache(self):
        """Loads DataFrames from Parquet and populates dicts."""
        tri = pd.read_parquet(self.cache_files["trigram"])
        bi_prev = pd.read_parquet(self.cache_files["bigram_prev"])
        bi_next = pd.read_parquet(self.cache_files["bigram_next"])
        uni = pd.read_parquet(self.cache_files["unigram"])

        self._populate_dicts(tri, bi_prev, bi_next, uni)

    def query(self, token, prev_token="<START>", next_token="<END>"):
        """
        Queries the hierarchical stats for a normalization prediction.

        Priority:
        1. Trigram (Exact context match)
        2. Bigram Prev (Left context match)
        3. Bigram Next (Right context match)
        4. Unigram (No context match) - Returns confidence score

        Args:
            token (str): The token to normalize.
            prev_token (str): The previous token in the sentence.
            next_token (str): The next token in the sentence.

        Returns:
            tuple: (prediction (str|None), confidence (float))
        """
        # 1. Trigram
        res = self.trigram_dict.get((prev_token, token, next_token))
        if res is not None:
            return res, 1.0

        # 2. Bigram Prev
        res = self.bigram_prev_dict.get((prev_token, token))
        if res is not None:
            return res, 1.0

        # 3. Bigram Next
        res = self.bigram_next_dict.get((token, next_token))
        if res is not None:
            return res, 1.0

        # 4. Unigram
        res_tuple = self.unigram_dict.get(token)
        if res_tuple is not None:
            pred, conf = res_tuple
            return pred, conf

        # OOV (Out of Vocabulary)
        return None, 0.0
