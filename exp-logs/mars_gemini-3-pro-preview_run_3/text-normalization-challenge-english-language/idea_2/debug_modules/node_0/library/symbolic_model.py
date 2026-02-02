import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data import load_metadata


class SymbolicMemory:
    """
    Symbolic "Head" solver using hierarchical N-gram lookup tables.
    Implements a cascade of Trigram -> Bigram -> Unigram lookups.
    """

    def __init__(self):
        # Lookup tables
        self.trigrams = {}  # (prev, curr, next) -> after
        self.bigrams_left = {}  # (prev, curr) -> after
        self.bigrams_right = {}  # (curr, next) -> after
        self.unigrams = {}  # curr -> after

    def fit(self, load_cached_data=True):
        """
        Builds the N-gram statistics from the training data.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed stats from disk.
        """
        # Define cache paths
        # We use Config.WORKING_DIR which is ./working/idea_2
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        tri_path = os.path.join(cache_dir, "stats_trigram.parquet")
        bi_l_path = os.path.join(cache_dir, "stats_bigram_left.parquet")
        bi_r_path = os.path.join(cache_dir, "stats_bigram_right.parquet")
        uni_path = os.path.join(cache_dir, "stats_unigram.parquet")

        # Check if all cache files exist
        all_cached = all(
            os.path.exists(p) for p in [tri_path, bi_l_path, bi_r_path, uni_path]
        )

        if load_cached_data and all_cached:
            print("Loading symbolic stats from cache...")
            self._load_from_cache(tri_path, bi_l_path, bi_r_path, uni_path)
        else:
            print("Computing symbolic stats from scratch...")
            self._compute_stats(tri_path, bi_l_path, bi_r_path, uni_path)

    def _compute_stats(self, tri_path, bi_l_path, bi_r_path, uni_path):
        """
        Computes N-gram stats from raw training data and saves to cache.
        """
        # Load full training data (no filtering)
        print("Loading training metadata...")
        df = load_metadata("train")

        # Ensure string types
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Generate Context (prev/next)
        # Logic replicated from library.data.process_data to ensure consistency
        # but applied to the full dataset (including PLAIN/PUNCT)
        print("Generating context columns for symbolic stats...")

        if "sentence_id" in df.columns:
            # Sort to ensure correct shifting
            df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

            sent_ids = df["sentence_id"]
            prev_sent_ids = sent_ids.shift(1)
            next_sent_ids = sent_ids.shift(-1)

            prev_series = df["before"].shift(1).fillna("")
            next_series = df["before"].shift(-1).fillna("")

            # Mask boundaries
            is_same_prev = sent_ids == prev_sent_ids
            is_same_next = sent_ids == next_sent_ids

            df["prev"] = np.where(is_same_prev, prev_series, "")
            df["next"] = np.where(is_same_next, next_series, "")
        else:
            # Fallback if no sentence_id (unlikely based on metadata)
            df["prev"] = df["before"].shift(1).fillna("")
            df["next"] = df["before"].shift(-1).fillna("")

        # Helper to compute best mapping (MLE)
        def process_and_save(group_cols, save_path):
            print(f"Aggregating stats for {group_cols}...")
            # Count occurrences of (context) -> after
            # We want the most frequent 'after' for each context

            # Groupby size is efficient
            counts = df.groupby(group_cols + ["after"]).size().reset_index(name="count")

            # Sort by count descending
            counts = counts.sort_values("count", ascending=False)

            # Drop duplicates on context columns, keeping the first (most frequent)
            best_mapping = counts.drop_duplicates(subset=group_cols, keep="first")

            # Drop the count column as we only need the mapping
            best_mapping = best_mapping.drop(columns=["count"])

            # Save to parquet
            print(f"Saving {len(best_mapping)} rules to {save_path}...")
            best_mapping.to_parquet(save_path, index=False)
            return best_mapping

        # 1. Trigrams: (prev, curr, next)
        df_tri = process_and_save(["prev", "before", "next"], tri_path)
        self._populate_dict(df_tri, ["prev", "before", "next"], self.trigrams)

        # 2. Left Bigrams: (prev, curr)
        df_bi_l = process_and_save(["prev", "before"], bi_l_path)
        self._populate_dict(df_bi_l, ["prev", "before"], self.bigrams_left)

        # 3. Right Bigrams: (curr, next)
        df_bi_r = process_and_save(["before", "next"], bi_r_path)
        self._populate_dict(df_bi_r, ["before", "next"], self.bigrams_right)

        # 4. Unigrams: (curr)
        df_uni = process_and_save(["before"], uni_path)
        self._populate_dict(df_uni, ["before"], self.unigrams)

        print("Symbolic stats computation complete.")

    def _load_from_cache(self, tri_path, bi_l_path, bi_r_path, uni_path):
        """Loads stats from parquet files into memory."""
        print(f"Loading Trigrams from {tri_path}...")
        self._populate_dict(
            pd.read_parquet(tri_path), ["prev", "before", "next"], self.trigrams
        )

        print(f"Loading Left Bigrams from {bi_l_path}...")
        self._populate_dict(
            pd.read_parquet(bi_l_path), ["prev", "before"], self.bigrams_left
        )

        print(f"Loading Right Bigrams from {bi_r_path}...")
        self._populate_dict(
            pd.read_parquet(bi_r_path), ["before", "next"], self.bigrams_right
        )

        print(f"Loading Unigrams from {uni_path}...")
        self._populate_dict(pd.read_parquet(uni_path), ["before"], self.unigrams)

    def _populate_dict(self, df, key_cols, target_dict):
        """
        Converts a DataFrame mapping into the target dictionary.
        """
        # Ensure all key columns are strings
        for col in key_cols:
            df[col] = df[col].astype(str)

        targets = df["after"].tolist()

        if len(key_cols) == 1:
            # Single key (Unigram): key is string
            keys = df[key_cols[0]].tolist()
            target_dict.update(zip(keys, targets))
        else:
            # Multiple keys: key is tuple
            # zip(*[col1, col2...]) creates tuples
            keys = zip(*[df[col] for col in key_cols])
            target_dict.update(zip(keys, targets))

    def query(self, prev, curr, next_tok):
        """
        Queries the symbolic memory with hierarchical fallback.

        Args:
            prev (str): Previous token text.
            curr (str): Current token text.
            next_tok (str): Next token text.

        Returns:
            str or None: The normalized text if found, else None.
        """
        # Ensure inputs are strings
        prev = str(prev)
        curr = str(curr)
        next_tok = str(next_tok)

        # 1. Trigram Check
        res = self.trigrams.get((prev, curr, next_tok))
        if res is not None:
            return res

        # 2. Left Bigram Check
        res = self.bigrams_left.get((prev, curr))
        if res is not None:
            return res

        # 3. Right Bigram Check
        res = self.bigrams_right.get((curr, next_tok))
        if res is not None:
            return res

        # 4. Unigram Check
        res = self.unigrams.get(curr)
        if res is not None:
            return res

        # 5. Fallback
        return None
