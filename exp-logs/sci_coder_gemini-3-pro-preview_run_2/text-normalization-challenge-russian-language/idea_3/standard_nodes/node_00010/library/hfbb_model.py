import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_factory import _add_context_and_filter


class HFBBEngine:
    """
    Tier 1: Granular Hierarchical Frequency-Based Backoff (HFBB) Engine.

    Implements a memory-based normalization engine using a strict backoff hierarchy:
    1. Trigram (prev, curr, next)
    2. Bigram (prev, curr)
    3. Bigram (curr, next)
    4. Unigram (curr)
    """

    def __init__(self):
        self.trigram_map = {}
        self.bigram_prev_map = {}
        self.bigram_next_map = {}
        self.unigram_map = {}
        self.is_fitted = False

    def fit(self, load_cached_data=True):
        """
        Builds the frequency maps from the training data or loads them from cache.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed maps from disk.
        """
        # Define cache paths
        cache_dir = Config.HFBB_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        tri_path = os.path.join(cache_dir, "trigram.parquet")
        bi_prev_path = os.path.join(cache_dir, "bigram_prev.parquet")
        bi_next_path = os.path.join(cache_dir, "bigram_next.parquet")
        uni_path = os.path.join(cache_dir, "unigram.parquet")

        paths = [tri_path, bi_prev_path, bi_next_path, uni_path]
        all_exist = all(os.path.exists(p) for p in paths)

        if load_cached_data and all_exist:
            print("Loading HFBB maps from cache...")
            self._load_maps(tri_path, bi_prev_path, bi_next_path, uni_path)
        else:
            print("Computing HFBB maps from training data...")
            # Load raw training data
            try:
                df = pd.read_csv(Config.TRAIN_CSV)
            except FileNotFoundError:
                print(f"Error: Training file {Config.TRAIN_CSV} not found.")
                return

            # Add context (prev/next). is_train=False ensures we keep ALL tokens (no semiotic filtering)
            # We need all tokens to build the full language statistics.
            # load_cached_data=False ensures we don't pick up a filtered cache from the Transformer step.
            df = _add_context_and_filter(df, is_train=False, load_cached_data=False)

            # Ensure strings to prevent type mismatch in lookups
            df["before"] = df["before"].fillna("").astype(str)
            df["after"] = df["after"].fillna("").astype(str)
            df["prev"] = df["prev"].fillna("").astype(str)
            df["next"] = df["next"].fillna("").astype(str)

            # 1. Unigrams
            print("Building Unigram map...")
            uni_df = self._build_stat_df(df, ["before"], "after")
            uni_df.to_parquet(uni_path, index=False)

            # 2. Bigrams (Prev)
            print("Building Bigram (Prev) map...")
            bi_prev_df = self._build_stat_df(df, ["prev", "before"], "after")
            bi_prev_df.to_parquet(bi_prev_path, index=False)

            # 3. Bigrams (Next)
            print("Building Bigram (Next) map...")
            bi_next_df = self._build_stat_df(df, ["before", "next"], "after")
            bi_next_df.to_parquet(bi_next_path, index=False)

            # 4. Trigrams
            print("Building Trigram map...")
            tri_df = self._build_stat_df(df, ["prev", "before", "next"], "after")
            tri_df.to_parquet(tri_path, index=False)

            # Load into memory
            self._load_maps(tri_path, bi_prev_path, bi_next_path, uni_path)

        self.is_fitted = True
        print("HFBB Engine fitted successfully.")

        # Optional: Evaluate on validation set to verify baseline performance
        self.evaluate()

    def _build_stat_df(self, df, group_cols, target_col):
        """
        Helper to compute the most frequent target value for a given grouping.
        """
        # Count occurrences of each combination
        # groupby size is faster than value_counts on full df
        cols = group_cols + [target_col]
        counts = df.groupby(cols).size().reset_index(name="count")

        # Sort by count descending
        counts = counts.sort_values("count", ascending=False)

        # Drop duplicates to keep the most frequent (the mode)
        best = counts.drop_duplicates(subset=group_cols)

        return best[cols]  # Keep only key columns and target

    def _load_maps(self, tri_path, bi_prev_path, bi_next_path, uni_path):
        """
        Loads parquet files into python dictionaries for O(1) lookup.
        """
        # Unigram: before -> after
        uni_df = pd.read_parquet(uni_path)
        self.unigram_map = dict(zip(uni_df["before"], uni_df["after"]))

        # Bigram Prev: (prev, before) -> after
        bi_prev_df = pd.read_parquet(bi_prev_path)
        self.bigram_prev_map = dict(
            zip(zip(bi_prev_df["prev"], bi_prev_df["before"]), bi_prev_df["after"])
        )

        # Bigram Next: (before, next) -> after
        bi_next_df = pd.read_parquet(bi_next_path)
        self.bigram_next_map = dict(
            zip(zip(bi_next_df["before"], bi_next_df["next"]), bi_next_df["after"])
        )

        # Trigram: (prev, before, next) -> after
        tri_df = pd.read_parquet(tri_path)
        self.trigram_map = dict(
            zip(zip(tri_df["prev"], tri_df["before"], tri_df["next"]), tri_df["after"])
        )

    def query(self, curr, prev, next_):
        """
        Queries the engine for a normalization using the backoff hierarchy.

        Args:
            curr (str): The token to normalize.
            prev (str): The preceding token.
            next_ (str): The succeeding token.

        Returns:
            tuple: (normalized_text, level)
                   level is one of ['trigram', 'bigram_prev', 'bigram_next', 'unigram', None]
                   If normalized_text is None, level is None.
        """
        if not self.is_fitted:
            raise RuntimeError("HFBBEngine must be fitted before querying.")

        curr = str(curr)
        prev = str(prev)
        next_ = str(next_)

        # 1. Trigram
        res = self.trigram_map.get((prev, curr, next_))
        if res is not None:
            return res, "trigram"

        # 2. Bigram Prev
        res = self.bigram_prev_map.get((prev, curr))
        if res is not None:
            return res, "bigram_prev"

        # 3. Bigram Next
        res = self.bigram_next_map.get((curr, next_))
        if res is not None:
            return res, "bigram_next"

        # 4. Unigram
        res = self.unigram_map.get(curr)
        if res is not None:
            return res, "unigram"

        return None, None

    def evaluate(self):
        """
        Evaluates the HFBB engine on the validation set.
        """
        print("Evaluating HFBB on Validation Set...")
        try:
            df_val = pd.read_csv(Config.VAL_CSV)
        except FileNotFoundError:
            print("Validation file not found. Skipping evaluation.")
            return

        # Prepare validation data with context
        df_val = _add_context_and_filter(df_val, is_train=False, load_cached_data=False)

        # Pre-convert columns to lists for speed
        l_prev = df_val["prev"].fillna("").astype(str).tolist()
        l_curr = df_val["before"].fillna("").astype(str).tolist()
        l_next = df_val["next"].fillna("").astype(str).tolist()
        actuals = df_val["after"].fillna("").astype(str).tolist()

        count_correct = 0
        total = len(df_val)

        # Iterate and query
        for p, c, n, act in zip(l_prev, l_curr, l_next, actuals):
            pred, _ = self.query(c, p, n)

            # If HFBB returns None, it means it hasn't seen this token.
            # In a full pipeline, this goes to Tier 2.
            # For pure HFBB evaluation, we fall back to Identity.
            if pred is None:
                pred = c

            if pred == act:
                count_correct += 1

        acc = count_correct / total
        print(f"HFBB Validation Accuracy: {acc}")
        return acc
