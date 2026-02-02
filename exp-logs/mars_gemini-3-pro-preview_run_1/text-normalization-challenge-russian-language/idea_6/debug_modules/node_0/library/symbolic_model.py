import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from library.config import Config, PAD_TOKEN
from library.utils import safe_save_dataframe, safe_load_dataframe, ensure_dir


class HierarchicalNgram:
    """
    Implements a hierarchical symbolic lookup system for text normalization.
    Prioritizes specific context (Trigram) over general context (Bigram/Unigram).
    """

    def __init__(self, config: Config):
        self.config = config
        # Stats containers: Key -> Normalized Text
        self.trigram_stats: Dict[Tuple[str, str, str], str] = {}
        self.bigram_stats: Dict[Tuple[str, str], str] = {}
        self.unigram_stats: Dict[str, str] = {}

    def build_stats(
        self, train_df: Optional[pd.DataFrame] = None, load_cached_data: bool = True
    ):
        """
        Constructs N-gram statistics from the training data or loads them from cache.

        Args:
            train_df: DataFrame containing 'sentence_id', 'before', 'after'.
            load_cached_data: If True, attempts to load from disk first.
        """
        # Define cache paths with run hash to prevent collisions
        run_hash = self.config.get_run_hash()
        tri_path = os.path.join(
            self.config.working_dir, f"trigram_stats_{run_hash}.parquet"
        )
        bi_path = os.path.join(
            self.config.working_dir, f"bigram_stats_{run_hash}.parquet"
        )
        uni_path = os.path.join(
            self.config.working_dir, f"unigram_stats_{run_hash}.parquet"
        )

        # 1. Attempt to load from cache
        if (
            load_cached_data
            and os.path.exists(tri_path)
            and os.path.exists(bi_path)
            and os.path.exists(uni_path)
        ):
            print("Loading cached N-gram statistics...")
            self._load_stats(tri_path, bi_path, uni_path)
            return

        # 2. Compute from scratch
        if train_df is None:
            raise ValueError(
                "train_df must be provided if cached data is not available."
            )

        print("Building N-gram statistics from scratch...")

        # Work on a copy to avoid modifying the original
        df = train_df.copy()

        # Ensure text columns are strings
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # --- Context Extraction ---
        # We need 'prev' and 'next' tokens for each row.
        # Vectorized shift is much faster than iteration.
        print("Extracting context windows...")

        # Shift
        df["prev"] = df["before"].shift(1).fillna(PAD_TOKEN)
        df["next"] = df["before"].shift(-1).fillna(PAD_TOKEN)

        # Handle Sentence Boundaries
        # If sentence_id[i] != sentence_id[i-1], then prev[i] is invalid (start of sentence)
        # If sentence_id[i] != sentence_id[i+1], then next[i] is invalid (end of sentence)
        sent_ids = df["sentence_id"].values

        # Check start of sentence (current != prev)
        # np.roll shifts elements; we compare id[i] with id[i-1]
        # roll(ids, 1) moves last to first, so we must handle index 0 explicitly
        prev_sent_ids = np.roll(sent_ids, 1)
        is_start = sent_ids != prev_sent_ids
        is_start[0] = True  # First row is always start

        # Check end of sentence (current != next)
        next_sent_ids = np.roll(sent_ids, -1)
        is_end = sent_ids != next_sent_ids
        is_end[-1] = True  # Last row is always end

        # Apply masks
        df.loc[is_start, "prev"] = PAD_TOKEN
        df.loc[is_end, "next"] = PAD_TOKEN

        # --- Aggregation ---
        # For each N-gram, we want the MOST FREQUENT 'after' value.

        # 3. Build Unigrams (curr -> after)
        print("Aggregating Unigrams...")
        uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Sort by count desc, then drop duplicates keeping the top one
        uni_best = uni_counts.sort_values("count", ascending=False).drop_duplicates(
            subset=["before"], keep="first"
        )
        safe_save_dataframe(uni_best, uni_path)

        # 4. Build Bigrams (prev, curr -> after)
        # Note: We focus on Left Context (prev, curr) as it's most predictive for reading flow
        print("Aggregating Bigrams...")
        bi_counts = (
            df.groupby(["prev", "before", "after"]).size().reset_index(name="count")
        )
        bi_best = bi_counts.sort_values("count", ascending=False).drop_duplicates(
            subset=["prev", "before"], keep="first"
        )
        safe_save_dataframe(bi_best, bi_path)

        # 5. Build Trigrams (prev, curr, next -> after)
        print("Aggregating Trigrams...")
        tri_counts = (
            df.groupby(["prev", "before", "next", "after"])
            .size()
            .reset_index(name="count")
        )
        tri_best = tri_counts.sort_values("count", ascending=False).drop_duplicates(
            subset=["prev", "before", "next"], keep="first"
        )
        safe_save_dataframe(tri_best, tri_path)

        # Load into memory
        self._load_stats(tri_path, bi_path, uni_path)

    def _load_stats(self, tri_path: str, bi_path: str, uni_path: str):
        """
        Loads the Parquet files and converts them to Python dictionaries.
        """
        tri_df = safe_load_dataframe(tri_path)
        bi_df = safe_load_dataframe(bi_path)
        uni_df = safe_load_dataframe(uni_path)

        print(
            f"Stats Loaded: Unigrams={len(uni_df)}, Bigrams={len(bi_df)}, Trigrams={len(tri_df)}"
        )

        # Convert to Dictionaries for O(1) lookup
        # Unigram: str -> str
        self.unigram_stats = dict(zip(uni_df["before"], uni_df["after"]))

        # Bigram: (str, str) -> str
        # We zip the key columns into tuples
        self.bigram_stats = dict(
            zip(zip(bi_df["prev"], bi_df["before"]), bi_df["after"])
        )

        # Trigram: (str, str, str) -> str
        self.trigram_stats = dict(
            zip(zip(tri_df["prev"], tri_df["before"], tri_df["next"]), tri_df["after"])
        )

    def get_trigram(self, prev: str, curr: str, next: str) -> Optional[str]:
        """Specific lookup for Trigram."""
        return self.trigram_stats.get((prev, curr, next))

    def get_bigram(self, prev: str, curr: str) -> Optional[str]:
        """Specific lookup for Bigram."""
        return self.bigram_stats.get((prev, curr))

    def get_unigram(self, curr: str) -> Optional[str]:
        """Specific lookup for Unigram."""
        return self.unigram_stats.get(curr)

    def query(self, prev: str, curr: str, next: str) -> str:
        """
        Executes the hierarchical fallback logic:
        1. Trigram (Exact context match)
        2. Bigram (Left context match)
        3. Unigram (Token match)
        4. Identity (Return original token)

        Args:
            prev: The previous token (or PAD_TOKEN)
            curr: The current token to normalize
            next: The next token (or PAD_TOKEN)

        Returns:
            The normalized string.
        """
        # 1. Trigram
        res = self.get_trigram(prev, curr, next)
        if res is not None:
            return res

        # 2. Bigram
        res = self.get_bigram(prev, curr)
        if res is not None:
            return res

        # 3. Unigram
        res = self.get_unigram(curr)
        if res is not None:
            return res

        # 4. Identity Fallback
        return curr
