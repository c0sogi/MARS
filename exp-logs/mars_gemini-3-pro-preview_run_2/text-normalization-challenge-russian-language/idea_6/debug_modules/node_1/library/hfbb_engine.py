import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from library.config import Config
from library.utils import set_seed, save_parquet_cache, load_parquet_cache


class HFBB:
    """
    Tier 1: Granular Memory Engine (HFBB)

    Implements a hierarchical statistical backoff model using:
    1. Trigram (Prev, Curr, Next)
    2. Bigram Prev (Prev, Curr)
    3. Bigram Next (Curr, Next)
    4. Unigram (Curr)

    Stores mappings based on Maximum Likelihood Estimation (most frequent target).
    """

    def __init__(self):
        self.trigram_map: Dict[Tuple[str, str, str], str] = {}
        self.bigram_prev_map: Dict[Tuple[str, str], str] = {}
        self.bigram_next_map: Dict[Tuple[str, str], str] = {}
        self.unigram_map: Dict[str, str] = {}

        # Define cache file paths
        self.cache_paths = {
            "trigram": os.path.join(Config.HFBB_CACHE_DIR, "trigram.parquet"),
            "bigram_prev": os.path.join(Config.HFBB_CACHE_DIR, "bigram_prev.parquet"),
            "bigram_next": os.path.join(Config.HFBB_CACHE_DIR, "bigram_next.parquet"),
            "unigram": os.path.join(Config.HFBB_CACHE_DIR, "unigram.parquet"),
        }

    def fit(
        self, df: Optional[pd.DataFrame] = None, load_cached_data: bool = True
    ) -> None:
        """
        Fits the HFBB model. Attempts to load from cache first.

        Args:
            df: Training dataframe containing 'sentence_id', 'before', 'after'.
                Required if cache is not used or missing.
            load_cached_data: If True, attempts to load pre-computed maps from disk.
        """
        set_seed()

        # Check if all cache files exist
        all_cache_exists = all(os.path.exists(p) for p in self.cache_paths.values())

        if load_cached_data and all_cache_exists:
            print(f"Loading HFBB maps from cache: {Config.HFBB_CACHE_DIR}")
            self._load_maps()
        else:
            if df is None:
                raise ValueError(
                    "Training DataFrame is required when cache is missing or load_cached_data=False."
                )

            print("Computing HFBB maps from scratch (this may take a moment)...")
            self._compute_maps(df)

            print(f"Saving HFBB maps to cache: {Config.HFBB_CACHE_DIR}")
            self._save_maps()

    def predict_token(
        self, token: str, prev_token: str = "<START>", next_token: str = "<END>"
    ) -> Optional[str]:
        """
        Predicts the normalized token using the hierarchical backoff strategy.

        Args:
            token: The current raw token.
            prev_token: The previous raw token (context).
            next_token: The next raw token (context).

        Returns:
            The normalized string if found, else None.
        """
        # 1. Trigram Check
        res = self.trigram_map.get((prev_token, token, next_token))
        if res is not None:
            return res

        # 2. Bigram (Prev) Check
        res = self.bigram_prev_map.get((prev_token, token))
        if res is not None:
            return res

        # 3. Bigram (Next) Check
        res = self.bigram_next_map.get((token, next_token))
        if res is not None:
            return res

        # 4. Unigram Check
        res = self.unigram_map.get(token)
        if res is not None:
            return res

        return None

    def _compute_maps(self, df: pd.DataFrame) -> None:
        """
        Computes n-gram statistics from the training data.
        """
        # Ensure we are working with strings
        df = df.copy()
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # Create context columns efficiently
        # We shift 'before' column. To prevent leakage across sentences, we check sentence_id boundaries.

        # Shift
        df["prev"] = df["before"].shift(1).fillna("<START>")
        df["next"] = df["before"].shift(-1).fillna("<END>")

        # Mask boundaries
        # If current sentence_id != prev sentence_id, then prev context is <START>
        is_start = df["sentence_id"] != df["sentence_id"].shift(1)
        df.loc[is_start, "prev"] = "<START>"

        # If current sentence_id != next sentence_id, then next context is <END>
        is_end = df["sentence_id"] != df["sentence_id"].shift(-1)
        df.loc[is_end, "next"] = "<END>"

        # Compute Maps using Maximum Likelihood Estimation
        # Strategy: Group by Keys + Target -> Count -> Sort Descending -> Drop Duplicates (keep top)

        print("  Computing Unigrams...")
        self.unigram_map = self._get_most_frequent(df, ["before"], "after")

        print("  Computing Bigrams (Prev)...")
        self.bigram_prev_map = self._get_most_frequent(df, ["prev", "before"], "after")

        print("  Computing Bigrams (Next)...")
        self.bigram_next_map = self._get_most_frequent(df, ["before", "next"], "after")

        print("  Computing Trigrams...")
        self.trigram_map = self._get_most_frequent(
            df, ["prev", "before", "next"], "after"
        )

    def _get_most_frequent(
        self, df: pd.DataFrame, group_cols: List[str], target_col: str
    ) -> Dict:
        """
        Helper to extract the most frequent target for a given set of grouping columns.
        """
        # Count occurrences of each (Group + Target) tuple
        counts = df.groupby(group_cols + [target_col]).size().reset_index(name="count")

        # Sort by count descending
        counts = counts.sort_values("count", ascending=False)

        # Drop duplicates on the grouping keys, keeping the one with highest count
        best = counts.drop_duplicates(subset=group_cols, keep="first")

        # Convert to dictionary
        if len(group_cols) == 1:
            return dict(zip(best[group_cols[0]], best[target_col]))
        else:
            # Create tuple keys
            keys = zip(*[best[col] for col in group_cols])
            return dict(zip(keys, best[target_col]))

    def _save_maps(self) -> None:
        """
        Saves the dictionaries to Parquet files.
        """
        # Unigram
        df_uni = pd.DataFrame(
            list(self.unigram_map.items()), columns=["curr", "target"]
        )
        save_parquet_cache(df_uni, self.cache_paths["unigram"])

        # Bigram Prev
        if self.bigram_prev_map:
            keys, values = zip(*self.bigram_prev_map.items())
            prevs, currs = zip(*keys)
            df_bi_prev = pd.DataFrame({"prev": prevs, "curr": currs, "target": values})
        else:
            df_bi_prev = pd.DataFrame(columns=["prev", "curr", "target"])
        save_parquet_cache(df_bi_prev, self.cache_paths["bigram_prev"])

        # Bigram Next
        if self.bigram_next_map:
            keys, values = zip(*self.bigram_next_map.items())
            currs, nexts = zip(*keys)
            df_bi_next = pd.DataFrame({"curr": currs, "next": nexts, "target": values})
        else:
            df_bi_next = pd.DataFrame(columns=["curr", "next", "target"])
        save_parquet_cache(df_bi_next, self.cache_paths["bigram_next"])

        # Trigram
        if self.trigram_map:
            keys, values = zip(*self.trigram_map.items())
            prevs, currs, nexts = zip(*keys)
            df_tri = pd.DataFrame(
                {"prev": prevs, "curr": currs, "next": nexts, "target": values}
            )
        else:
            df_tri = pd.DataFrame(columns=["prev", "curr", "next", "target"])
        save_parquet_cache(df_tri, self.cache_paths["trigram"])

    def _load_maps(self) -> None:
        """
        Loads dictionaries from Parquet files.
        """
        # Unigram
        df_uni = load_parquet_cache(self.cache_paths["unigram"])
        self.unigram_map = dict(zip(df_uni["curr"], df_uni["target"]))

        # Bigram Prev
        df_bi_prev = load_parquet_cache(self.cache_paths["bigram_prev"])
        self.bigram_prev_map = dict(
            zip(zip(df_bi_prev["prev"], df_bi_prev["curr"]), df_bi_prev["target"])
        )

        # Bigram Next
        df_bi_next = load_parquet_cache(self.cache_paths["bigram_next"])
        self.bigram_next_map = dict(
            zip(zip(df_bi_next["curr"], df_bi_next["next"]), df_bi_next["target"])
        )

        # Trigram
        df_tri = load_parquet_cache(self.cache_paths["trigram"])
        self.trigram_map = dict(
            zip(zip(df_tri["prev"], df_tri["curr"], df_tri["next"]), df_tri["target"])
        )
