import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Optional, Tuple, Dict, List
from library.config import Config
from library.data_processor import SOS_TOKEN, EOS_TOKEN


class NgramMemory:
    """
    Symbolic Memory component of the Hybrid Neuro-Symbolic System.
    Stores Hierarchical N-gram statistics (Trigram, Bigram, Unigram)
    to handle deterministic and frequent token normalizations.
    """

    def __init__(self, config: Config):
        self.config = config
        self.trigrams: Dict[Tuple[str, str, str], str] = {}
        self.bigrams: Dict[Tuple[str, str], str] = {}
        self.unigrams: Dict[str, str] = {}

        # Define cache paths within the versioned directory
        self.trigram_path = os.path.join(
            self.config.version_dir, "trigram_stats.parquet"
        )
        self.bigram_path = os.path.join(self.config.version_dir, "bigram_stats.parquet")
        self.unigram_path = os.path.join(
            self.config.version_dir, "unigram_stats.parquet"
        )

    def build_stats(self, load_cached_data: bool = True):
        """
        Builds or loads N-gram statistics from the training data.

        Args:
            load_cached_data (bool): If True, attempts to load from disk cache first.
        """
        # 1. Try Loading from Cache
        if load_cached_data and self._check_cache_exists():
            print("Loading N-gram stats from cache...")
            self._load_stats()
            return

        # 2. Compute from Scratch
        print("Computing N-gram stats from training data...")

        # Load training metadata
        train_meta_path = os.path.join(self.config.metadata_dir, "train.csv")
        if not os.path.exists(train_meta_path):
            raise FileNotFoundError(f"Training metadata not found at {train_meta_path}")

        # Read data with specific dtypes for efficiency
        df = pd.read_csv(
            train_meta_path,
            dtype={
                "sentence_id": "int32",
                "token_id": "int32",
                "before": "object",
                "after": "object",
            },
        )

        # Handle potential NaNs (though metadata script handles them, extra safety)
        df["before"] = df["before"].fillna("")
        df["after"] = df["after"].fillna("")

        # Sort by sentence_id and token_id to ensure correct sequence order
        df = df.sort_values(["sentence_id", "token_id"])

        # Group by sentence_id
        grouped = df.groupby("sentence_id")

        # Initialize Counters
        # Trigram: (prev, curr, next) -> Counter(after)
        trigram_counts = defaultdict(Counter)
        # Bigram: (prev, curr) -> Counter(after)
        bigram_counts = defaultdict(Counter)
        # Unigram: (curr) -> Counter(after)
        unigram_counts = defaultdict(Counter)

        print(f"Processing {len(grouped)} sentences for N-gram extraction...")

        # Iterate over sentences to extract contexts
        for _, group in grouped:
            tokens_in = group["before"].tolist()
            tokens_out = group["after"].tolist()
            seq_len = len(tokens_in)

            for i in range(seq_len):
                curr_tok = tokens_in[i]
                target_norm = tokens_out[i]

                # Context extraction with padding
                prev_tok = tokens_in[i - 1] if i > 0 else SOS_TOKEN
                next_tok = tokens_in[i + 1] if i < seq_len - 1 else EOS_TOKEN

                # Update Counters
                trigram_counts[(prev_tok, curr_tok, next_tok)][target_norm] += 1
                bigram_counts[(prev_tok, curr_tok)][target_norm] += 1
                unigram_counts[curr_tok][target_norm] += 1

        print("Aggregating and filtering stats (removing identity mappings)...")

        # Helper to process counters into DataFrames
        def process_counters(counter_dict, context_len):
            data = []
            for context, counts in counter_dict.items():
                # Get most frequent normalization
                best_norm, _ = counts.most_common(1)[0]

                # Determine raw token from context to check for identity
                if context_len == 3:
                    # (prev, curr, next)
                    raw_token = context[1]
                elif context_len == 2:
                    # (prev, curr)
                    raw_token = context[1]
                else:
                    # curr
                    raw_token = context

                # Optimization: Only store if normalization changes the text
                # If before == after, we rely on the Identity fallback in the pipeline
                if best_norm != raw_token:
                    if context_len == 3:
                        data.append(
                            {
                                "prev": context[0],
                                "curr": context[1],
                                "next": context[2],
                                "norm": best_norm,
                            }
                        )
                    elif context_len == 2:
                        data.append(
                            {"prev": context[0], "curr": context[1], "norm": best_norm}
                        )
                    else:
                        data.append({"curr": context, "norm": best_norm})

            # Define columns explicitly to handle empty data cases safely
            if context_len == 3:
                cols = ["prev", "curr", "next", "norm"]
            elif context_len == 2:
                cols = ["prev", "curr", "norm"]
            else:
                cols = ["curr", "norm"]

            return pd.DataFrame(data, columns=cols)

        # Process stats
        df_trigram = process_counters(trigram_counts, 3)
        df_bigram = process_counters(bigram_counts, 2)
        df_unigram = process_counters(unigram_counts, 1)

        print(
            f"Stats collected: Trigrams={len(df_trigram)}, Bigrams={len(df_bigram)}, Unigrams={len(df_unigram)}"
        )

        # Save to Cache
        self._save_parquet(df_trigram, self.trigram_path)
        self._save_parquet(df_bigram, self.bigram_path)
        self._save_parquet(df_unigram, self.unigram_path)

        # Load into memory
        self._load_stats()

    def _check_cache_exists(self) -> bool:
        """Checks if all required stat files exist."""
        return (
            os.path.exists(self.trigram_path)
            and os.path.exists(self.bigram_path)
            and os.path.exists(self.unigram_path)
        )

    def _save_parquet(self, df: pd.DataFrame, path: str):
        """Saves DataFrame to Parquet."""
        df.to_parquet(path, index=False)

    def _load_stats(self):
        """Loads stats from Parquet files into memory dicts."""
        # Trigrams
        if os.path.exists(self.trigram_path):
            df = pd.read_parquet(self.trigram_path)
            if not df.empty:
                self.trigrams = {
                    (p, c, n): norm
                    for p, c, n, norm in zip(
                        df["prev"], df["curr"], df["next"], df["norm"]
                    )
                }

        # Bigrams
        if os.path.exists(self.bigram_path):
            df = pd.read_parquet(self.bigram_path)
            if not df.empty:
                self.bigrams = {
                    (p, c): norm
                    for p, c, norm in zip(df["prev"], df["curr"], df["norm"])
                }

        # Unigrams
        if os.path.exists(self.unigram_path):
            df = pd.read_parquet(self.unigram_path)
            if not df.empty:
                self.unigrams = {c: norm for c, norm in zip(df["curr"], df["norm"])}

    def query_trigram(
        self, prev_tok: str, curr_tok: str, next_tok: str
    ) -> Optional[str]:
        """
        Queries the Trigram memory.
        Context: (Previous, Current, Next)
        Returns: Normalized text if found, else None.
        """
        return self.trigrams.get((prev_tok, curr_tok, next_tok))

    def query_bigram(self, prev_tok: str, curr_tok: str) -> Optional[str]:
        """
        Queries the Bigram memory.
        Context: (Previous, Current)
        Returns: Normalized text if found, else None.
        """
        return self.bigrams.get((prev_tok, curr_tok))

    def query_unigram(self, curr_tok: str) -> Optional[str]:
        """
        Queries the Unigram memory.
        Context: (Current)
        Returns: Normalized text if found, else None.
        """
        return self.unigrams.get(curr_tok)
