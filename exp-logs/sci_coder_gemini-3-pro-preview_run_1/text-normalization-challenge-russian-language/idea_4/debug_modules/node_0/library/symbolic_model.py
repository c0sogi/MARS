import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from library.utils import get_or_compute, load_metadata


class NgramLookup:
    """
    Symbolic model component that uses hierarchical N-gram statistics
    (Trigram -> Bigram -> Unigram) to normalize tokens.
    Acts as the 'Head' solver in the hybrid architecture.
    """

    def __init__(self, config):
        self.config = config
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        # Path for caching the computed statistics
        self.stats_path = os.path.join(
            config.working_dir, f"ngram_stats_{config.config_hash}.npy"
        )

    def fit(self, load_cached_data=True):
        """
        Computes or loads N-gram statistics from the training data.

        Args:
            load_cached_data (bool): If True, attempts to load from disk first.
        """

        def _compute_stats():
            print("Computing N-gram statistics from training data...")
            # Load training data using the utility to ensure correct dtypes
            df = load_metadata(self.config.train_file)

            # Sort by sentence_id and token_id to ensure correct sequence order
            if "token_id" in df.columns:
                df = df.sort_values(["sentence_id", "token_id"])

            # Group by sentence_id to reconstruct full sentences (context)
            # Aggregating into lists allows for efficient iteration
            print("Grouping data by sentence...")
            grouped = df.groupby("sentence_id")[["before", "after"]].agg(list)

            # Initialize counters for each hierarchy level
            uni_counts = defaultdict(Counter)
            bi_counts = defaultdict(Counter)
            tri_counts = defaultdict(Counter)

            print("Iterating sentences to build N-grams...")
            # Iterate over sentences
            # grouped['before'] is a Series of lists, same for 'after'
            # zip allows iterating them together
            count_sentences = 0
            for tokens, targets in zip(grouped["before"], grouped["after"]):
                seq_len = len(tokens)
                for i in range(seq_len):
                    # Ensure tokens are strings
                    curr_t = str(tokens[i])
                    target = str(targets[i])

                    # Define Context
                    # Use special tokens for boundaries
                    prev_t = str(tokens[i - 1]) if i > 0 else "<s>"
                    next_t = str(tokens[i + 1]) if i < seq_len - 1 else "</s>"

                    # 1. Unigram: curr -> target
                    uni_counts[curr_t][target] += 1

                    # 2. Bigram: (prev, curr) -> target
                    # Captures immediate left context (e.g., "Level" -> "5")
                    bi_counts[(prev_t, curr_t)][target] += 1

                    # 3. Trigram: (prev, curr, next) -> target
                    # Captures both sides (e.g., disambiguating based on following noun)
                    tri_counts[(prev_t, curr_t, next_t)][target] += 1

                count_sentences += 1

            print(f"Processed {count_sentences} sentences.")
            print("Reducing statistics to maximum likelihood targets...")

            # Helper function to extract the most frequent target for each key
            def get_best_mappings(counts_dict):
                # k is the n-gram key, c is the Counter of observed targets
                # most_common(1) returns [(value, count)]
                return {k: c.most_common(1)[0][0] for k, c in counts_dict.items()}

            final_uni = get_best_mappings(uni_counts)
            final_bi = get_best_mappings(bi_counts)
            final_tri = get_best_mappings(tri_counts)

            return (final_uni, final_bi, final_tri)

        # Use the utility to handle caching logic (load if exists, else compute and save)
        stats = get_or_compute(
            self.stats_path, _compute_stats, load_cached_data=load_cached_data
        )

        # Unpack the loaded data
        # np.save/load with object arrays can wrap the tuple in different ways
        if isinstance(stats, np.ndarray):
            if stats.shape == ():
                # 0-d array wrapping the tuple
                stats = stats.item()
            elif len(stats) == 3:
                # 1-d array or unpacked tuple
                pass

        self.unigrams, self.bigrams, self.trigrams = stats
        print(
            f"N-gram Lookup Ready: {len(self.unigrams)} Unigrams, {len(self.bigrams)} Bigrams, {len(self.trigrams)} Trigrams"
        )

    def get_normalization(self, curr, prev=None, next=None):
        """
        Predicts the normalized text using hierarchical backoff.

        Args:
            curr (str): The token to normalize.
            prev (str, optional): The previous token. Defaults to "<s>" if None.
            next (str, optional): The next token. Defaults to "</s>" if None.

        Returns:
            str or None: The normalized text if found in lookup, else None.
        """
        # Normalize inputs to strings to match keys
        curr = str(curr)
        prev = str(prev) if prev is not None else "<s>"
        next = str(next) if next is not None else "</s>"

        # 1. Trigram Lookup (Most Specific)
        # Key: (prev, curr, next)
        res = self.trigrams.get((prev, curr, next))
        if res is not None:
            return res

        # 2. Bigram Lookup
        # Key: (prev, curr)
        res = self.bigrams.get((prev, curr))
        if res is not None:
            return res

        # 3. Unigram Lookup (Least Specific)
        # Key: curr
        res = self.unigrams.get(curr)
        if res is not None:
            return res

        # 4. Fallback (Signal to use Neural Model)
        return None
