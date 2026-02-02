import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


class HierarchicalBackoff:
    """
    Tier 1: Confidence-Aware Granular Memory.
    Implements a 4-level hierarchical backoff strategy (Trigram -> Bigram Prev -> Bigram Next -> Unigram).
    Stores confidence scores for Unigrams to act as a gate for the Tier 2 Neural Network.
    """

    def __init__(self):
        # In-memory lookup tables
        # Structure:
        #   unigram: {current_token: (prediction, confidence)}
        #   bigram_prev: {(prev_token, current_token): prediction}
        #   bigram_next: {(current_token, next_token): prediction}
        #   trigram: {(prev_token, current_token, next_token): prediction}
        self.unigram = {}
        self.bigram_prev = {}
        self.bigram_next = {}
        self.trigram = {}

    def fit(self, load_cached_data: bool = True):
        """
        Constructs the statistical tables from the training data.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed tables
                                     from the cache directory defined in Config.
        """
        set_seed(Config.SEED)

        # Define cache paths
        cache_dir = os.path.join(Config.CACHE_DIR, "hfbb")
        os.makedirs(cache_dir, exist_ok=True)

        files = {
            "unigram": os.path.join(cache_dir, "unigram.parquet"),
            "bigram_prev": os.path.join(cache_dir, "bigram_prev.parquet"),
            "bigram_next": os.path.join(cache_dir, "bigram_next.parquet"),
            "trigram": os.path.join(cache_dir, "trigram.parquet"),
        }

        # 1. Attempt to load from cache
        if load_cached_data and all(os.path.exists(f) for f in files.values()):
            print(f"Loading HFBB tables from cache: {cache_dir}")
            self._load_tables(files)
            return

        # 2. Compute from scratch
        print("Computing HFBB tables from scratch (this may take a few minutes)...")

        # Load training data
        # We use the metadata split to ensure we are training on the correct subset
        df = pd.read_csv(Config.TRAIN_DATA)

        # Data cleaning: Ensure strict string types to prevent mixed-type errors
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Sort by sentence and token ID to ensure correct context extraction
        if "sentence_id" in df.columns and "token_id" in df.columns:
            df = df.sort_values(["sentence_id", "token_id"])

        # --- Context Generation (Vectorized) ---
        print("Generating context windows...")

        # Shift 'before' column to get previous and next tokens
        df["prev"] = df["before"].shift(1)
        df["next"] = df["before"].shift(-1)

        # Handle Sentence Boundaries
        # If sentence_id changes, the context is broken.
        # We use vectorized comparison to detect boundaries.

        # Start of sentence: current sentence_id != prev sentence_id
        start_mask = df["sentence_id"] != df["sentence_id"].shift(1)
        # The very first row is always a start
        start_mask.iloc[0] = True
        # Apply sentinel
        df.loc[start_mask, "prev"] = "<START>"
        # Fill any remaining NaNs (e.g. first row if not caught)
        df["prev"] = df["prev"].fillna("<START>")

        # End of sentence: current sentence_id != next sentence_id
        end_mask = df["sentence_id"] != df["sentence_id"].shift(-1)
        # The very last row is always an end
        end_mask.iloc[-1] = True
        # Apply sentinel
        df.loc[end_mask, "next"] = "<END>"
        df["next"] = df["next"].fillna("<END>")

        # --- Build Unigram Table (with Confidence) ---
        print("Building Unigram table...")
        # Count occurrences of (before -> after)
        uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Count total occurrences of (before)
        uni_totals = df.groupby("before").size().reset_index(name="total")

        # Find the mode (most frequent mapping)
        uni_counts = uni_counts.sort_values("count", ascending=False)
        uni_modes = uni_counts.drop_duplicates(subset=["before"], keep="first")

        # Calculate confidence: count(mode) / count(total)
        uni_final = pd.merge(uni_modes, uni_totals, on="before")
        uni_final["confidence"] = uni_final["count"] / uni_final["total"]

        # Save
        uni_final[["before", "after", "confidence"]].to_parquet(files["unigram"])

        # --- Build Bigram Prev Table ---
        print("Building Bigram (Prev) table...")
        # Group by (prev, before) -> after
        bi_prev_counts = (
            df.groupby(["prev", "before", "after"]).size().reset_index(name="count")
        )
        bi_prev_counts = bi_prev_counts.sort_values("count", ascending=False)
        bi_prev_modes = bi_prev_counts.drop_duplicates(
            subset=["prev", "before"], keep="first"
        )
        bi_prev_modes[["prev", "before", "after"]].to_parquet(files["bigram_prev"])

        # --- Build Bigram Next Table ---
        print("Building Bigram (Next) table...")
        # Group by (before, next) -> after
        bi_next_counts = (
            df.groupby(["before", "next", "after"]).size().reset_index(name="count")
        )
        bi_next_counts = bi_next_counts.sort_values("count", ascending=False)
        bi_next_modes = bi_next_counts.drop_duplicates(
            subset=["before", "next"], keep="first"
        )
        bi_next_modes[["before", "next", "after"]].to_parquet(files["bigram_next"])

        # --- Build Trigram Table ---
        print("Building Trigram table...")
        # Group by (prev, before, next) -> after
        tri_counts = (
            df.groupby(["prev", "before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        tri_counts = tri_counts.sort_values("count", ascending=False)
        tri_modes = tri_counts.drop_duplicates(
            subset=["prev", "before", "next"], keep="first"
        )
        tri_modes[["prev", "before", "next", "after"]].to_parquet(files["trigram"])

        # Load the newly computed tables into memory
        self._load_tables(files)
        print("HFBB fitting complete.")

    def _load_tables(self, files):
        """
        Loads Parquet files into in-memory dictionaries for O(1) lookup.
        """
        # Unigram
        df_uni = pd.read_parquet(files["unigram"])
        self.unigram = {
            k: (v, c)
            for k, v, c in zip(df_uni["before"], df_uni["after"], df_uni["confidence"])
        }

        # Bigram Prev
        df_bi_prev = pd.read_parquet(files["bigram_prev"])
        self.bigram_prev = {
            (p, b): a
            for p, b, a in zip(
                df_bi_prev["prev"], df_bi_prev["before"], df_bi_prev["after"]
            )
        }

        # Bigram Next
        df_bi_next = pd.read_parquet(files["bigram_next"])
        self.bigram_next = {
            (b, n): a
            for b, n, a in zip(
                df_bi_next["before"], df_bi_next["next"], df_bi_next["after"]
            )
        }

        # Trigram
        df_tri = pd.read_parquet(files["trigram"])
        self.trigram = {
            (p, b, n): a
            for p, b, n, a in zip(
                df_tri["prev"], df_tri["before"], df_tri["next"], df_tri["after"]
            )
        }

    def query(
        self, before: str, prev_token: str = "<START>", next_token: str = "<END>"
    ):
        """
        Queries the hierarchical model for a normalization prediction.

        Args:
            before (str): The token to normalize.
            prev_token (str): The previous token in the sentence.
            next_token (str): The next token in the sentence.

        Returns:
            tuple: (prediction, confidence, source_level)
                - prediction (str or None): The normalized text.
                - confidence (float): 0.0 to 1.0.
                - source_level (str): 'trigram', 'bigram_prev', 'bigram_next', 'unigram', or 'none'.
        """
        # 1. Trigram Check (Highest Specificity)
        res = self.trigram.get((prev_token, before, next_token))
        if res is not None:
            # Context-specific matches are treated as 100% confident for this tier
            return res, 1.0, "trigram"

        # 2. Bigram Prev Check
        res = self.bigram_prev.get((prev_token, before))
        if res is not None:
            return res, 1.0, "bigram_prev"

        # 3. Bigram Next Check
        res = self.bigram_next.get((before, next_token))
        if res is not None:
            return res, 1.0, "bigram_next"

        # 4. Unigram Check (Lowest Specificity, requires Confidence Check)
        res_tuple = self.unigram.get(before)
        if res_tuple is not None:
            pred, conf = res_tuple
            return pred, conf, "unigram"

        # 5. No Match
        return None, 0.0, "none"
