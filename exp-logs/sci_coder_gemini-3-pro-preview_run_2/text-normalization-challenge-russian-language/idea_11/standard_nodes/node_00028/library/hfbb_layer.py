import pandas as pd
from typing import Optional, Tuple
from library.config import Config
from library.data_manager import build_hfbb_stats
from library.utils import timer


class HFBBModel:
    """
    Tier 1: Confidence-Aware Memory Engine.
    Implements a hierarchical backoff strategy using statistical N-gram models.
    Acts as a high-precision filter before delegating complex cases to the neural network.
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initialize the HFBB model by loading statistics and building lookup maps.

        Args:
            load_cached_data (bool): Whether to load stats from cache or recompute.
        """
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

        # Load raw statistics (DataFrames)
        with timer("Loading HFBB Statistics"):
            # This function from data_manager handles caching of the parquet files
            stats = build_hfbb_stats(load_cached_data=load_cached_data)

        # Build optimized in-memory lookup maps
        with timer("Building HFBB Lookup Maps"):
            self._build_maps(stats)

    def _build_maps(self, stats: dict):
        """
        Converts pandas DataFrames into python dictionaries for O(1) inference.
        """
        # 1. Trigram Map: (prev_1, before, next_1) -> after
        df_tri = stats["trigram"]
        # Ensure string types to avoid lookup failures and zip for speed
        p1 = df_tri["prev_1"].astype(str)
        curr = df_tri["before"].astype(str)
        n1 = df_tri["next_1"].astype(str)
        tgt = df_tri["after"].astype(str)
        self.trigram_map = dict(zip(zip(p1, curr, n1), tgt))

        # 2. Bigram Prev Map: (prev_1, before) -> after
        df_bip = stats["bigram_prev"]
        p1 = df_bip["prev_1"].astype(str)
        curr = df_bip["before"].astype(str)
        tgt = df_bip["after"].astype(str)
        self.bigram_prev_map = dict(zip(zip(p1, curr), tgt))

        # 3. Bigram Next Map: (before, next_1) -> after
        df_bin = stats["bigram_next"]
        curr = df_bin["before"].astype(str)
        n1 = df_bin["next_1"].astype(str)
        tgt = df_bin["after"].astype(str)
        self.bigram_next_map = dict(zip(zip(curr, n1), tgt))

        # 4. Unigram Map: before -> (after, confidence)
        df_uni = stats["unigram"]
        curr = df_uni["before"].astype(str)
        tgt = df_uni["after"].astype(str)
        conf = df_uni["confidence"].astype(float)
        self.unigram_map = dict(zip(curr, zip(tgt, conf)))

        print(
            f"HFBB Maps Built: Trigram={len(self.trigram_map)}, "
            f"BigramPrev={len(self.bigram_prev_map)}, "
            f"BigramNext={len(self.bigram_next_map)}, "
            f"Unigram={len(self.unigram_map)}"
        )

    def query(self, before: str, prev_1: str = "", next_1: str = "") -> Optional[str]:
        """
        Query the hierarchical model for a normalization prediction.

        Args:
            before (str): The token to normalize.
            prev_1 (str): The previous token context.
            next_1 (str): The next token context.

        Returns:
            Optional[str]: The predicted normalized text, or None if no confident match found.
        """
        # Ensure inputs are strings
        before = str(before)
        prev_1 = str(prev_1)
        next_1 = str(next_1)

        # Step 1: Trigram Check
        # Specificity: High (Context Left & Right)
        # Resolves highly context-dependent ambiguities
        if (prev_1, before, next_1) in self.trigram_map:
            return self.trigram_map[(prev_1, before, next_1)]

        # Step 2: Bigram (Prev) Check
        # Specificity: Medium (Context Left)
        if (prev_1, before) in self.bigram_prev_map:
            return self.bigram_prev_map[(prev_1, before)]

        # Step 3: Bigram (Next) Check
        # Specificity: Medium (Context Right)
        if (before, next_1) in self.bigram_next_map:
            return self.bigram_next_map[(before, next_1)]

        # Step 4: Unigram Check with Confidence Gating
        # Specificity: Low (No Context)
        # Only returns if the model is highly confident (e.g., > 99%)
        if before in self.unigram_map:
            prediction, confidence = self.unigram_map[before]
            if confidence >= self.confidence_threshold:
                return prediction

        # Fallback to Tier 2 (Neural Model)
        # This occurs if the token is unseen OR if it is ambiguous (low confidence unigram)
        return None
