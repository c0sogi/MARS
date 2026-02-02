import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from library.config import Config


class SymbolicMemory:
    """
    Implements the 'Head' solver using hierarchical N-gram statistics.

    This class builds and queries a prioritized hierarchy of hash maps:
    Trigrams -> Bigrams (Left/Right) -> Unigrams. It serves as the first
    line of defense in the normalization pipeline, handling frequent and
    context-dependent tokens via exact memorization.
    """

    def __init__(self):
        self.trigram_stats: Dict[Tuple[str, str, str], str] = {}
        self.bigram_left_stats: Dict[Tuple[str, str], str] = {}
        self.bigram_right_stats: Dict[Tuple[str, str], str] = {}
        self.unigram_stats: Dict[str, str] = {}

        # Special tokens for context boundaries
        self.token_start = "<start>"
        self.token_end = "<end>"

    def build_stats(self, load_cached_data: bool = True):
        """
        Constructs the N-gram statistics from the training data.

        Args:
            load_cached_data: If True, attempts to load pre-computed stats
                              from the cache directory defined in Config.
        """
        # Define paths
        paths = {
            "trigram": Config.TRIGRAM_STATS_PATH,
            "bigram_left": Config.BIGRAM_LEFT_STATS_PATH,
            "bigram_right": Config.BIGRAM_RIGHT_STATS_PATH,
            "unigram": Config.UNIGRAM_STATS_PATH,
        }

        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print("Loading symbolic stats from cache...")
            self._load_stats(paths)
        else:
            print("Computing symbolic stats from scratch...")
            self._compute_and_save_stats(paths)

        print(
            f"Symbolic Memory Loaded: {len(self.unigram_stats)} unigrams, "
            f"{len(self.bigram_left_stats)} left-bigrams, "
            f"{len(self.bigram_right_stats)} right-bigrams, "
            f"{len(self.trigram_stats)} trigrams."
        )

    def _compute_and_save_stats(self, paths: Dict[str, str]):
        """
        Computes N-gram statistics from the raw training data and saves them.
        """
        # Ensure output directory exists
        os.makedirs(Config.STATS_DIR, exist_ok=True)

        # Load training data
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {Config.TRAIN_DATA_PATH}"
            )

        df = pd.read_parquet(Config.TRAIN_DATA_PATH)

        # Optional: Subsample for debugging if configured
        if Config.MAX_TRAIN_SAMPLES is not None:
            df = df.head(Config.MAX_TRAIN_SAMPLES).copy()

        # Ensure string types
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # Sort to ensure correct context extraction
        df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

        # Generate Context Columns
        # Shift 'before' to get prev and next tokens
        df["prev_before"] = df["before"].shift(1)
        df["next_before"] = df["before"].shift(-1)

        # Shift 'sentence_id' to detect boundaries
        df["prev_sent"] = df["sentence_id"].shift(1)
        df["next_sent"] = df["sentence_id"].shift(-1)

        # Handle Boundaries: If sentence ID changes, context is <start>/<end>
        # We use numpy where for speed
        df["prev_before"] = np.where(
            df["prev_sent"] == df["sentence_id"], df["prev_before"], self.token_start
        )
        df["next_before"] = np.where(
            df["next_sent"] == df["sentence_id"], df["next_before"], self.token_end
        )

        # Fill NaNs at the very beginning/end of the dataframe
        df["prev_before"] = df["prev_before"].fillna(self.token_start)
        df["next_before"] = df["next_before"].fillna(self.token_end)

        # --- Aggregation Logic ---
        # Helper to get most frequent 'after' for given keys
        def get_top_stats(groupby_cols):
            # Group by keys + target, count, sort descending, take top 1
            # Using pandas optimized operations
            counts = (
                df.groupby(groupby_cols + ["after"]).size().reset_index(name="count")
            )
            # Sort by count (desc) then by after (lexicographical for stability)
            counts = counts.sort_values(["count", "after"], ascending=[False, True])
            # Drop duplicates to keep only the top 1 for each key combination
            top_stats = counts.drop_duplicates(subset=groupby_cols)
            return top_stats.drop(columns=["count"])

        print("Aggregating Unigrams...")
        unigram_df = get_top_stats(["before"])
        unigram_df.to_parquet(paths["unigram"], index=False)
        self.unigram_stats = dict(zip(unigram_df["before"], unigram_df["after"]))

        print("Aggregating Left Bigrams...")
        bigram_l_df = get_top_stats(["prev_before", "before"])
        bigram_l_df.to_parquet(paths["bigram_left"], index=False)
        self.bigram_left_stats = dict(
            zip(
                zip(bigram_l_df["prev_before"], bigram_l_df["before"]),
                bigram_l_df["after"],
            )
        )

        print("Aggregating Right Bigrams...")
        bigram_r_df = get_top_stats(["before", "next_before"])
        bigram_r_df.to_parquet(paths["bigram_right"], index=False)
        self.bigram_right_stats = dict(
            zip(
                zip(bigram_r_df["before"], bigram_r_df["next_before"]),
                bigram_r_df["after"],
            )
        )

        print("Aggregating Trigrams...")
        trigram_df = get_top_stats(["prev_before", "before", "next_before"])
        trigram_df.to_parquet(paths["trigram"], index=False)
        self.trigram_stats = dict(
            zip(
                zip(
                    trigram_df["prev_before"],
                    trigram_df["before"],
                    trigram_df["next_before"],
                ),
                trigram_df["after"],
            )
        )

    def _load_stats(self, paths: Dict[str, str]):
        """
        Loads pre-computed statistics from Parquet files into memory.
        """
        # Load Unigrams
        unigram_df = pd.read_parquet(paths["unigram"])
        self.unigram_stats = dict(zip(unigram_df["before"], unigram_df["after"]))

        # Load Left Bigrams
        bigram_l_df = pd.read_parquet(paths["bigram_left"])
        self.bigram_left_stats = dict(
            zip(
                zip(bigram_l_df["prev_before"], bigram_l_df["before"]),
                bigram_l_df["after"],
            )
        )

        # Load Right Bigrams
        bigram_r_df = pd.read_parquet(paths["bigram_right"])
        self.bigram_right_stats = dict(
            zip(
                zip(bigram_r_df["before"], bigram_r_df["next_before"]),
                bigram_r_df["after"],
            )
        )

        # Load Trigrams
        trigram_df = pd.read_parquet(paths["trigram"])
        self.trigram_stats = dict(
            zip(
                zip(
                    trigram_df["prev_before"],
                    trigram_df["before"],
                    trigram_df["next_before"],
                ),
                trigram_df["after"],
            )
        )

    def query(
        self,
        token: str,
        prev_token: Optional[str] = None,
        next_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Queries the hierarchical memory for a normalization.

        Priority:
        1. Trigram (prev, curr, next)
        2. Bigram Left (prev, curr)
        3. Bigram Right (curr, next)
        4. Unigram (curr)

        Args:
            token: The current token to normalize.
            prev_token: The previous token in the sentence (None if start).
            next_token: The next token in the sentence (None if end).

        Returns:
            The normalized string if found, else None.
        """
        # Normalize context inputs
        p_tok = str(prev_token) if prev_token is not None else self.token_start
        n_tok = str(next_token) if next_token is not None else self.token_end
        c_tok = str(token)

        # 1. Check Trigram
        trigram_key = (p_tok, c_tok, n_tok)
        if trigram_key in self.trigram_stats:
            return self.trigram_stats[trigram_key]

        # 2. Check Bigram Left
        bigram_l_key = (p_tok, c_tok)
        if bigram_l_key in self.bigram_left_stats:
            return self.bigram_left_stats[bigram_l_key]

        # 3. Check Bigram Right
        bigram_r_key = (c_tok, n_tok)
        if bigram_r_key in self.bigram_right_stats:
            return self.bigram_right_stats[bigram_r_key]

        # 4. Check Unigram
        if c_tok in self.unigram_stats:
            return self.unigram_stats[c_tok]

        # Not found in symbolic memory
        return None
