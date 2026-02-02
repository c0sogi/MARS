import pandas as pd
import numpy as np
from library.config import Config
from library.data_manager import build_symbolic_stats


class SymbolicModel:
    """
    Implements the 'Head' (Symbolic Memory) and 'Gate' (Heuristic Router)
    of the Hybrid Neuro-Symbolic architecture.
    """

    def __init__(self, df_train=None, load_cached_data=True):
        """
        Initialize the symbolic model by loading or building N-gram statistics.

        Args:
            df_train (pd.DataFrame, optional): Training data to build stats from if cache is missing.
            load_cached_data (bool): Whether to attempt loading stats from cache.
        """
        print("Initializing Symbolic Model...")
        # Load stats using the data manager utility
        self.stats = build_symbolic_stats(df_train, load_cached_data=load_cached_data)

        # Unpack dictionaries for faster lookup during inference
        # Priority: Trigram -> Bigram_Left -> Bigram_Right -> Unigram
        self.trigram = self.stats.get("trigram", {})
        self.bigram_left = self.stats.get("bigram_left", {})
        self.bigram_right = self.stats.get("bigram_right", {})
        self.unigram = self.stats.get("unigram", {})

        print(f"Symbolic Model Ready.")
        print(f"  Trigrams: {len(self.trigram)}")
        print(f"  Left Bigrams: {len(self.bigram_left)}")
        print(f"  Right Bigrams: {len(self.bigram_right)}")
        print(f"  Unigrams: {len(self.unigram)}")

    def resolve(self, prev_tok, curr_tok, next_tok):
        """
        Queries the symbolic memory hierarchy for a match.

        Args:
            prev_tok: The token preceding the current one.
            curr_tok: The token to be normalized.
            next_tok: The token following the current one.

        Returns:
            str: The normalized text if found, else None.
        """
        # Normalize inputs to strings to match dictionary keys
        # The data manager uses fillna("") for context, so we use empty string for None/NaN context
        # For current token, we cast to string.

        p = str(prev_tok) if prev_tok is not None and not pd.isna(prev_tok) else ""
        c = str(curr_tok) if curr_tok is not None and not pd.isna(curr_tok) else ""
        n = str(next_tok) if next_tok is not None and not pd.isna(next_tok) else ""

        # 1. Trigram Lookup (prev, curr, next)
        res = self.trigram.get((p, c, n))
        if res is not None:
            return res

        # 2. Left Bigram Lookup (prev, curr)
        res = self.bigram_left.get((p, c))
        if res is not None:
            return res

        # 3. Right Bigram Lookup (curr, next)
        res = self.bigram_right.get((c, n))
        if res is not None:
            return res

        # 4. Unigram Lookup (curr)
        res = self.unigram.get(c)
        if res is not None:
            return res

        # No match found in symbolic memory
        return None

    def heuristic_gate(self, curr_tok):
        """
        Applies heuristic rules to filter simple OOV tokens before neural processing.

        Args:
            curr_tok: The token to check.

        Returns:
            str: The token itself if it passes the heuristic (Identity), else None.
        """
        s = str(curr_tok)

        # Heuristic: If the token is purely alphabetic, it is likely a regular word
        # that was simply rare (OOV) and doesn't need normalization (e.g. proper nouns).
        # We return the identity mapping.
        if s.isalpha():
            return s

        return None
