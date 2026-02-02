import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


class HFBBEngine:
    """
    Tier 1: The Granular Memory Engine (Word-Level).
    Implements a 4-level backoff hierarchy:
    Trigram -> Bigram (Prev) -> Bigram (Next) -> Unigram.

    This engine memorizes the most frequent normalization for token contexts
    seen in the training data.
    """

    def __init__(self):
        self.unigram_map = {}
        self.bigram_prev_map = {}
        self.bigram_next_map = {}
        self.trigram_map = {}

        # Define cache file paths
        self.cache_files = {
            "unigram": os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet"),
            "bigram_prev": os.path.join(Config.HFBB_CACHE_DIR, "bigram_prev.parquet"),
            "bigram_next": os.path.join(Config.HFBB_CACHE_DIR, "bigram_next.parquet"),
            "trigram": os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet"),
        }

    def fit(self, train_df, load_cached_data=True):
        """
        Constructs the frequency maps from the training data.

        Args:
            train_df (pd.DataFrame): The training data containing 'before', 'after',
                                     'sentence_id', 'token_id'.
            load_cached_data (bool): If True, attempts to load maps from disk
                                     instead of recomputing.
        """
        set_seed()

        # Check if all cache files exist
        all_cached = all(os.path.exists(p) for p in self.cache_files.values())

        if load_cached_data and all_cached:
            print("HFBB: Loading cached frequency maps...")
            self._load_cache()
        else:
            print("HFBB: Computing frequency maps from scratch...")
            self._compute_maps(train_df)
            print("HFBB: Saving maps to cache...")
            self._save_cache()

        print(
            f"HFBB: Ready. "
            f"Unigrams: {len(self.unigram_map)}, "
            f"Bigrams(P): {len(self.bigram_prev_map)}, "
            f"Bigrams(N): {len(self.bigram_next_map)}, "
            f"Trigrams: {len(self.trigram_map)}"
        )

    def query(self, token, prev_word, next_word):
        """
        Queries the hierarchy for a normalized form.

        Args:
            token (str): The current token (raw).
            prev_word (str): The previous token (raw).
            next_word (str): The next token (raw).

        Returns:
            str or None: The normalized text if found, else None.
        """
        # 1. Trigram Check
        tri_key = (prev_word, token, next_word)
        if tri_key in self.trigram_map:
            return self.trigram_map[tri_key]

        # 2. Bigram (Prev) Check
        bi_prev_key = (prev_word, token)
        if bi_prev_key in self.bigram_prev_map:
            return self.bigram_prev_map[bi_prev_key]

        # 3. Bigram (Next) Check
        bi_next_key = (token, next_word)
        if bi_next_key in self.bigram_next_map:
            return self.bigram_next_map[bi_next_key]

        # 4. Unigram Check
        if token in self.unigram_map:
            return self.unigram_map[token]

        return None

    def _compute_maps(self, df):
        """
        Internal method to compute statistics using vectorized Pandas operations.
        """
        # Ensure data is sorted to correctly identify prev/next
        # Although metadata is likely sorted, we enforce it.
        df = df.sort_values(["sentence_id", "token_id"]).copy()

        # Generate context columns
        # We use groupby sentence_id to ensure we don't bleed context across sentences
        print("HFBB: Generating context columns...")
        df["prev_word"] = (
            df.groupby("sentence_id")["before"].shift(1).fillna(Config.PAD_TOKEN)
        )
        df["next_word"] = (
            df.groupby("sentence_id")["before"].shift(-1).fillna(Config.PAD_TOKEN)
        )

        # Helper to get mode (most frequent 'after') for a given set of grouping keys
        def get_mode_map(group_cols):
            # Count occurrences of each combination
            counts = df.groupby(group_cols + ["after"]).size().reset_index(name="count")
            # Sort by count descending
            counts = counts.sort_values("count", ascending=False)
            # Drop duplicates keeping the first (most frequent)
            best = counts.drop_duplicates(subset=group_cols)
            return best

        # 1. Unigram
        print("HFBB: Computing Unigrams...")
        uni_df = get_mode_map(["before"])
        self.unigram_map = dict(zip(uni_df["before"], uni_df["after"]))
        self._uni_df_cache = uni_df[["before", "after"]]  # Store for saving

        # 2. Bigram Prev
        print("HFBB: Computing Bigrams (Prev)...")
        bi_p_df = get_mode_map(["prev_word", "before"])
        self.bigram_prev_map = dict(
            zip(zip(bi_p_df["prev_word"], bi_p_df["before"]), bi_p_df["after"])
        )
        self._bi_p_df_cache = bi_p_df[["prev_word", "before", "after"]]

        # 3. Bigram Next
        print("HFBB: Computing Bigrams (Next)...")
        bi_n_df = get_mode_map(["before", "next_word"])
        self.bigram_next_map = dict(
            zip(zip(bi_n_df["before"], bi_n_df["next_word"]), bi_n_df["after"])
        )
        self._bi_n_df_cache = bi_n_df[["before", "next_word", "after"]]

        # 4. Trigram
        print("HFBB: Computing Trigrams...")
        tri_df = get_mode_map(["prev_word", "before", "next_word"])
        self.trigram_map = dict(
            zip(
                zip(tri_df["prev_word"], tri_df["before"], tri_df["next_word"]),
                tri_df["after"],
            )
        )
        self._tri_df_cache = tri_df[["prev_word", "before", "next_word", "after"]]

    def _save_cache(self):
        """Saves computed maps to Parquet."""
        Config.setup_directories()

        self._uni_df_cache.to_parquet(self.cache_files["unigram"], index=False)
        self._bi_p_df_cache.to_parquet(self.cache_files["bigram_prev"], index=False)
        self._bi_n_df_cache.to_parquet(self.cache_files["bigram_next"], index=False)
        self._tri_df_cache.to_parquet(self.cache_files["trigram"], index=False)

        # Clear temp dfs to free memory
        del self._uni_df_cache
        del self._bi_p_df_cache
        del self._bi_n_df_cache
        del self._tri_df_cache

    def _load_cache(self):
        """Loads maps from Parquet."""
        # Unigram
        df = pd.read_parquet(self.cache_files["unigram"])
        self.unigram_map = dict(zip(df["before"], df["after"]))

        # Bigram Prev
        df = pd.read_parquet(self.cache_files["bigram_prev"])
        self.bigram_prev_map = dict(
            zip(zip(df["prev_word"], df["before"]), df["after"])
        )

        # Bigram Next
        df = pd.read_parquet(self.cache_files["bigram_next"])
        self.bigram_next_map = dict(
            zip(zip(df["before"], df["next_word"]), df["after"])
        )

        # Trigram
        df = pd.read_parquet(self.cache_files["trigram"])
        self.trigram_map = dict(
            zip(zip(df["prev_word"], df["before"], df["next_word"]), df["after"])
        )
