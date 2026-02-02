import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import save_data, load_data


class HFBBModel:
    """
    Hierarchical Frequency-Based Backoff (HFBB) Model.

    This model implements a text normalization strategy based on n-gram statistics
    collected from the training corpus. It uses a backoff hierarchy:
    Trigram -> Bigram (Prev) -> Bigram (Next) -> Unigram -> Identity.
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

    def fit(self, train_df=None, load_cached_data=True):
        """
        Fits the model by aggregating statistics from the training data or loading from cache.

        Args:
            train_df (pd.DataFrame, optional): The training data containing 'sentence_id',
                                               'token_id', 'before', 'after'.
            load_cached_data (bool): If True, attempts to load pre-computed stats from disk.
        """
        # Ensure cache directory exists
        os.makedirs(Config.HFBB_CACHE_DIR, exist_ok=True)

        # Check if all cache files exist
        all_cache_exists = all(os.path.exists(p) for p in self.cache_files.values())

        if load_cached_data and all_cache_exists:
            print("Loading HFBB statistics from cache...")
            self._load_maps_from_cache()
            return

        if train_df is None:
            raise ValueError(
                "Cache missing (or load_cached_data=False) and no training data provided."
            )

        print("Computing HFBB statistics from training data...")
        self._compute_and_cache_stats(train_df)
        print("HFBB statistics computed and cached.")

    def _compute_and_cache_stats(self, train_df):
        """
        Computes n-gram statistics from the dataframe and saves them to parquet.
        """
        # working on a copy to avoid modifying original
        df = train_df.copy()

        # Ensure string types and handle NaNs
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # Ensure data is sorted to correctly determine context
        # We assume sentence_id and token_id are present and sortable
        df.sort_values(["sentence_id", "token_id"], inplace=True)

        # Generate Context Columns
        # Shift to get previous and next tokens
        df["prev_before"] = df["before"].shift(1).fillna("<start>")
        df["next_before"] = df["before"].shift(-1).fillna("<end>")

        # Shift sentence_id to detect boundaries
        df["prev_sent"] = df["sentence_id"].shift(1)
        df["next_sent"] = df["sentence_id"].shift(-1)

        # Handle Sentence Boundaries
        # If prev sentence != current sentence, prev_token is <start>
        # We compare with !=, which handles NaN correctly (NaN != value is True)
        start_mask = df["prev_sent"] != df["sentence_id"]
        df.loc[start_mask, "prev_before"] = "<start>"

        # If next sentence != current sentence, next_token is <end>
        end_mask = df["next_sent"] != df["sentence_id"]
        df.loc[end_mask, "next_before"] = "<end>"

        # Helper to get best mapping (mode)
        def get_best_mapping(dataframe, group_cols):
            # Group by context + target, count occurrences
            counts = (
                dataframe.groupby(group_cols + ["after"])
                .size()
                .reset_index(name="count")
            )
            # Sort by count descending
            counts.sort_values("count", ascending=False, inplace=True)
            # Drop duplicates keeping the first (most frequent)
            best = counts.drop_duplicates(subset=group_cols)
            return best[group_cols + ["after"]]

        # 1. Unigram: before -> after
        print("  Computing Unigrams...")
        df_uni = get_best_mapping(df, ["before"])
        save_data(df_uni, self.cache_files["unigram"])
        self.unigram_map = dict(zip(df_uni["before"], df_uni["after"]))

        # 2. Bigram Prev: (prev, before) -> after
        print("  Computing Bigrams (Prev)...")
        df_bi_prev = get_best_mapping(df, ["prev_before", "before"])
        save_data(df_bi_prev, self.cache_files["bigram_prev"])
        self.bigram_prev_map = dict(
            zip(
                zip(df_bi_prev["prev_before"], df_bi_prev["before"]),
                df_bi_prev["after"],
            )
        )

        # 3. Bigram Next: (before, next) -> after
        print("  Computing Bigrams (Next)...")
        df_bi_next = get_best_mapping(df, ["before", "next_before"])
        save_data(df_bi_next, self.cache_files["bigram_next"])
        self.bigram_next_map = dict(
            zip(
                zip(df_bi_next["before"], df_bi_next["next_before"]),
                df_bi_next["after"],
            )
        )

        # 4. Trigram: (prev, before, next) -> after
        print("  Computing Trigrams...")
        df_tri = get_best_mapping(df, ["prev_before", "before", "next_before"])
        save_data(df_tri, self.cache_files["trigram"])
        self.trigram_map = dict(
            zip(
                zip(df_tri["prev_before"], df_tri["before"], df_tri["next_before"]),
                df_tri["after"],
            )
        )

    def _load_maps_from_cache(self):
        """
        Loads statistics from parquet files into dictionaries.
        """
        # Unigram
        df_uni = load_data(self.cache_files["unigram"])
        self.unigram_map = dict(zip(df_uni["before"], df_uni["after"]))

        # Bigram Prev
        df_bi_prev = load_data(self.cache_files["bigram_prev"])
        self.bigram_prev_map = dict(
            zip(
                zip(df_bi_prev["prev_before"], df_bi_prev["before"]),
                df_bi_prev["after"],
            )
        )

        # Bigram Next
        df_bi_next = load_data(self.cache_files["bigram_next"])
        self.bigram_next_map = dict(
            zip(
                zip(df_bi_next["before"], df_bi_next["next_before"]),
                df_bi_next["after"],
            )
        )

        # Trigram
        df_tri = load_data(self.cache_files["trigram"])
        self.trigram_map = dict(
            zip(
                zip(df_tri["prev_before"], df_tri["before"], df_tri["next_before"]),
                df_tri["after"],
            )
        )

    def predict(self, token, prev_token=None, next_token=None):
        """
        Predicts the normalized text for a given token using the backoff hierarchy.

        Args:
            token (str): The token to normalize.
            prev_token (str, optional): The preceding token. Defaults to "<start>".
            next_token (str, optional): The succeeding token. Defaults to "<end>".

        Returns:
            str: The normalized text.
        """
        # Normalize inputs
        t = str(token)
        p = str(prev_token) if prev_token is not None else "<start>"
        n = str(next_token) if next_token is not None else "<end>"

        # 1. Trigram Check
        tri_key = (p, t, n)
        if tri_key in self.trigram_map:
            return self.trigram_map[tri_key]

        # 2. Bigram Prev Check
        bi_prev_key = (p, t)
        if bi_prev_key in self.bigram_prev_map:
            return self.bigram_prev_map[bi_prev_key]

        # 3. Bigram Next Check
        bi_next_key = (t, n)
        if bi_next_key in self.bigram_next_map:
            return self.bigram_next_map[bi_next_key]

        # 4. Unigram Check
        if t in self.unigram_map:
            return self.unigram_map[t]

        # 5. Identity (Backoff to self)
        return t
