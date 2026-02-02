import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_utils import process_context


class SymbolicMemory:
    """
    A high-precision symbolic memory model that utilizes hierarchical hash maps
    (Trigrams, Bigrams, Unigrams) to resolve text normalization deterministically
    based on training data statistics.

    Hierarchy:
    1. Trigrams (prev, curr, next)
    2. Left Bigrams (prev, curr)
    3. Right Bigrams (curr, next)
    4. Unigrams (curr)
    """

    def __init__(self):
        self.trigram_map = {}
        self.bigram_left_map = {}
        self.bigram_right_map = {}
        self.unigram_map = {}

        self.stats_dir = Config.STATS_CACHE_DIR
        os.makedirs(self.stats_dir, exist_ok=True)

    def fit(self, load_cached_data=True):
        """
        Aggregates training data into hierarchical maps using Maximum Likelihood Estimation.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed stats from disk.
        """
        # Define paths for cached stats
        tri_path = os.path.join(self.stats_dir, "trigram_stats.parquet")
        bi_left_path = os.path.join(self.stats_dir, "bigram_left_stats.parquet")
        bi_right_path = os.path.join(self.stats_dir, "bigram_right_stats.parquet")
        uni_path = os.path.join(self.stats_dir, "unigram_stats.parquet")

        # Check if all cache files exist
        cache_exists = (
            os.path.exists(tri_path)
            and os.path.exists(bi_left_path)
            and os.path.exists(bi_right_path)
            and os.path.exists(uni_path)
        )

        if load_cached_data and cache_exists:
            print("Loading symbolic stats from cache...")
            self._load_stats(tri_path, bi_left_path, bi_right_path, uni_path)
            return

        print("Computing symbolic stats from scratch (MLE)...")

        # Load full training data
        # We use the metadata file directly to avoid the soft-filtering applied in data_utils.load_data
        if not os.path.exists(Config.TRAIN_META_PATH):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_META_PATH}"
            )

        df = pd.read_parquet(Config.TRAIN_META_PATH)

        # Ensure string types
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # Add context (prev, next)
        # We must ensure the dataframe is sorted by sentence_id, token_id before processing context
        if "sentence_id" in df.columns and "token_id" in df.columns:
            df = df.sort_values(["sentence_id", "token_id"])

        df = process_context(df)

        # Helper function to compute mode (most frequent mapping)
        def compute_mode(data, group_cols):
            # 1. Count frequencies
            # We group by input features + target
            counts = (
                data.groupby(group_cols + ["after"]).size().reset_index(name="count")
            )

            # 2. Sort to identify the mode
            # Primary sort: count (descending)
            # Secondary sort: after (ascending) for deterministic tie-breaking
            sort_order = [True] * len(group_cols) + [False, True]
            counts = counts.sort_values(
                by=group_cols + ["count", "after"], ascending=sort_order
            )

            # 3. Drop duplicates to keep the top 1 (mode) for each group
            modes = counts.drop_duplicates(subset=group_cols, keep="first")

            # Return just the mapping columns
            return modes[group_cols + ["after"]]

        # 1. Trigrams: (prev, before, next) -> after
        print("Aggregating Trigrams...")
        tri_df = compute_mode(df, ["prev", "before", "next"])
        tri_df.to_parquet(tri_path, index=False)

        # 2. Left Bigrams: (prev, before) -> after
        print("Aggregating Left Bigrams...")
        bi_left_df = compute_mode(df, ["prev", "before"])
        bi_left_df.to_parquet(bi_left_path, index=False)

        # 3. Right Bigrams: (before, next) -> after
        print("Aggregating Right Bigrams...")
        bi_right_df = compute_mode(df, ["before", "next"])
        bi_right_df.to_parquet(bi_right_path, index=False)

        # 4. Unigrams: (before) -> after
        print("Aggregating Unigrams...")
        uni_df = compute_mode(df, ["before"])
        uni_df.to_parquet(uni_path, index=False)

        # Load into memory
        self._load_stats(tri_path, bi_left_path, bi_right_path, uni_path)

    def _load_stats(self, tri_path, bi_left_path, bi_right_path, uni_path):
        """Loads parquet files into memory as dictionaries."""

        # Trigrams
        df = pd.read_parquet(tri_path)
        self.trigram_map = dict(
            zip(zip(df["prev"], df["before"], df["next"]), df["after"])
        )

        # Left Bigrams
        df = pd.read_parquet(bi_left_path)
        self.bigram_left_map = dict(zip(zip(df["prev"], df["before"]), df["after"]))

        # Right Bigrams
        df = pd.read_parquet(bi_right_path)
        self.bigram_right_map = dict(zip(zip(df["before"], df["next"]), df["after"]))

        # Unigrams
        df = pd.read_parquet(uni_path)
        self.unigram_map = dict(zip(df["before"], df["after"]))

        print(
            f"Symbolic Memory Loaded: {len(self.trigram_map)} trigrams, "
            f"{len(self.bigram_left_map)} left-bigrams, "
            f"{len(self.bigram_right_map)} right-bigrams, "
            f"{len(self.unigram_map)} unigrams."
        )

    def predict(self, token, prev_token="", next_token=""):
        """
        Queries the hierarchical memory to normalize a token.

        Priority:
        1. Trigram (prev, token, next)
        2. Left Bigram (prev, token)
        3. Right Bigram (token, next)
        4. Unigram (token)

        Returns:
            str or None: The normalized text if found, else None.
        """
        # Ensure inputs are strings
        token = str(token)
        prev_token = str(prev_token)
        next_token = str(next_token)

        # 1. Trigram
        res = self.trigram_map.get((prev_token, token, next_token))
        if res is not None:
            return res

        # 2. Left Bigram
        res = self.bigram_left_map.get((prev_token, token))
        if res is not None:
            return res

        # 3. Right Bigram
        res = self.bigram_right_map.get((token, next_token))
        if res is not None:
            return res

        # 4. Unigram
        res = self.unigram_map.get(token)
        if res is not None:
            return res

        return None
