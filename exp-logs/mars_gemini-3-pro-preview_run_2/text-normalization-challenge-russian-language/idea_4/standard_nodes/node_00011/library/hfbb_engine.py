import os
import pandas as pd
import numpy as np
from library.config import Config


class HFBBModel:
    """
    Tier 1: Granular Hierarchical Frequency-Based Backoff (HFBB) Model.

    This model memorizes mappings from context (Trigram, Bigram, Unigram) to normalized text.
    It acts as the first line of defense, handling known tokens and phrases with high precision.
    """

    def __init__(self):
        self.unigram_map = {}
        self.bigram_prev_map = {}
        self.bigram_next_map = {}
        self.trigram_map = {}

    def fit(self, load_cached_data: bool = True, max_rows: int = None):
        """
        Fits the HFBB model using the training data.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed maps from disk.
            max_rows (int, optional): Limit the number of training rows (for debugging).
        """
        # Define cache paths
        cache_files = {
            "unigram": os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet"),
            "bigram_prev": os.path.join(Config.HFBB_CACHE_DIR, "bigram_prev.parquet"),
            "bigram_next": os.path.join(Config.HFBB_CACHE_DIR, "bigram_next.parquet"),
            "trigram": os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet"),
        }

        # Check if all cache files exist
        all_cached = all(os.path.exists(p) for p in cache_files.values())

        if load_cached_data and all_cached and max_rows is None:
            print("Loading HFBB models from cache...")
            self._load_from_cache(cache_files)
        else:
            print("Computing HFBB models from scratch...")
            self._compute_and_cache(cache_files, max_rows)

    def get_normalization(
        self, token: str, prev_token: str = "<START>", next_token: str = "<END>"
    ) -> str:
        """
        Retrieves the normalization for a token based on its context using the strict hierarchy.

        Hierarchy: Trigram -> Bigram (Prev) -> Bigram (Next) -> Unigram

        Args:
            token (str): The current token to normalize.
            prev_token (str): The previous token in the sentence.
            next_token (str): The next token in the sentence.

        Returns:
            str: The normalized text if found, else None.
        """
        # 1. Trigram: (prev, current, next)
        res = self.trigram_map.get((prev_token, token, next_token))
        if res is not None:
            return res

        # 2. Bigram Prev: (prev, current)
        res = self.bigram_prev_map.get((prev_token, token))
        if res is not None:
            return res

        # 3. Bigram Next: (current, next)
        res = self.bigram_next_map.get((token, next_token))
        if res is not None:
            return res

        # 4. Unigram: (current)
        res = self.unigram_map.get(token)
        if res is not None:
            return res

        return None

    def _compute_and_cache(self, cache_files, max_rows):
        """Computes statistics from raw data and saves to cache."""
        # Ensure output directory exists
        os.makedirs(Config.HFBB_CACHE_DIR, exist_ok=True)

        # Load training data
        print(f"Reading training data from {Config.TRAIN_DATA}...")
        df = pd.read_csv(Config.TRAIN_DATA, nrows=max_rows)

        # Preprocessing
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Generate Context (Prev/Next) respecting sentence boundaries
        print("Generating context windows...")
        s_ids = df["sentence_id"].values
        tokens = df["before"].values

        # Shift tokens
        prev_tokens = np.roll(tokens, 1)
        next_tokens = np.roll(tokens, -1)

        # Shift sentence IDs to detect boundaries
        prev_s_ids = np.roll(s_ids, 1)
        next_s_ids = np.roll(s_ids, -1)

        # Handle boundaries
        # 1. Array edges
        prev_tokens[0] = "<START>"
        next_tokens[-1] = "<END>"

        # 2. Sentence changes
        # If current sentence ID != prev sentence ID, then prev token is <START>
        start_mask = s_ids != prev_s_ids
        prev_tokens[start_mask] = "<START>"

        # If current sentence ID != next sentence ID, then next token is <END>
        end_mask = s_ids != next_s_ids
        next_tokens[end_mask] = "<END>"

        df["prev"] = prev_tokens
        df["next"] = next_tokens

        # Compute and Cache Maps

        # 1. Unigram
        print("Building Unigram map...")
        uni_df = self._get_mode_map(df, ["before"], "after")
        if max_rows is None:  # Only save cache if full training
            uni_df.to_parquet(cache_files["unigram"])
        self.unigram_map = self._df_to_dict(uni_df, ["before"])

        # 2. Bigram Prev
        print("Building Bigram Prev map...")
        bi_prev_df = self._get_mode_map(df, ["prev", "before"], "after")
        if max_rows is None:
            bi_prev_df.to_parquet(cache_files["bigram_prev"])
        self.bigram_prev_map = self._df_to_dict(bi_prev_df, ["prev", "before"])

        # 3. Bigram Next
        print("Building Bigram Next map...")
        bi_next_df = self._get_mode_map(df, ["before", "next"], "after")
        if max_rows is None:
            bi_next_df.to_parquet(cache_files["bigram_next"])
        self.bigram_next_map = self._df_to_dict(bi_next_df, ["before", "next"])

        # 4. Trigram
        print("Building Trigram map...")
        tri_df = self._get_mode_map(df, ["prev", "before", "next"], "after")
        if max_rows is None:
            tri_df.to_parquet(cache_files["trigram"])
        self.trigram_map = self._df_to_dict(tri_df, ["prev", "before", "next"])

        print("HFBB fitting complete.")

    def _load_from_cache(self, cache_files):
        """Loads maps from parquet files."""
        print("Loading Unigram map...")
        self.unigram_map = self._df_to_dict(
            pd.read_parquet(cache_files["unigram"]), ["before"]
        )

        print("Loading Bigram Prev map...")
        self.bigram_prev_map = self._df_to_dict(
            pd.read_parquet(cache_files["bigram_prev"]), ["prev", "before"]
        )

        print("Loading Bigram Next map...")
        self.bigram_next_map = self._df_to_dict(
            pd.read_parquet(cache_files["bigram_next"]), ["before", "next"]
        )

        print("Loading Trigram map...")
        self.trigram_map = self._df_to_dict(
            pd.read_parquet(cache_files["trigram"]), ["prev", "before", "next"]
        )

    def _get_mode_map(self, df, key_cols, target_col):
        """
        Calculates the mode (most frequent) target value for each key combination.
        """
        # Count occurrences: Group by keys + target -> count
        counts = df.groupby(key_cols + [target_col]).size().reset_index(name="count")

        # Sort by count descending
        counts = counts.sort_values("count", ascending=False)

        # Drop duplicates on keys, keeping the first (highest count)
        mode_df = counts.drop_duplicates(subset=key_cols, keep="first")

        return mode_df[key_cols + [target_col]]

    def _df_to_dict(self, df, key_cols):
        """Converts a DataFrame to a dictionary for fast O(1) lookup."""
        keys = df[key_cols].values
        targets = df["after"].values

        if len(key_cols) == 1:
            # Scalar keys
            return dict(zip(keys.flatten(), targets))
        else:
            # Tuple keys
            # map(tuple, keys) is efficient for converting rows to tuples
            return dict(zip(map(tuple, keys), targets))
