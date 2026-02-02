import os
from library.utils import load_npy
from library.data_processing import build_ngram_statistics, load_and_group_data


class SymbolicLookup:
    """
    The 'Head' solver: A hierarchical N-gram lookup table.
    Memorizes frequent patterns (Trigram -> Bigram -> Unigram) to handle
    common words, punctuation, and fixed phrases efficiently.
    """

    def __init__(self, train_grouped_df=None, load_cached=True):
        """
        Initialize the symbolic lookup table.

        Args:
            train_grouped_df (pd.DataFrame, optional): The grouped training data.
                                                       If None and stats not cached, will load from disk.
            load_cached (bool): Whether to attempt loading stats from cache.
        """
        self.stats = None

        # 1. Try loading from cache first to avoid data loading overhead
        # We check manually here to avoid loading the dataframe if stats exist
        if load_cached:
            self.stats = load_npy("ngram_stats.npy")

        # 2. If not found in cache, we must build it
        if self.stats is None:
            # We need the training data to build stats.
            # If not provided, load it using the library function.
            if train_grouped_df is None:
                # print("Loading training data for SymbolicLookup...")
                train_grouped_df = load_and_group_data("train", load_cached_data=True)

            # Build (and save) stats using the library function
            self.stats = build_ngram_statistics(
                train_grouped_df, load_cached_data=load_cached
            )

    def query(self, prev_word, curr_word, next_word):
        """
        Query the N-gram stats for a normalization.

        Implements the hierarchy:
        1. Trigram (Prev, Curr, Next)
        2. Bigram (Prev, Curr)
        3. Unigram (Curr)

        Args:
            prev_word (str): The previous token text (or "<start>").
            curr_word (str): The current token text.
            next_word (str): The next token text (or "<end>").

        Returns:
            str or None: The normalized text if found, else None.
        """
        if self.stats is None:
            return None

        # 1. Trigram Lookup (Specific Context)
        # Checks if the exact sequence (prev, curr, next) has been seen
        trigram_key = (prev_word, curr_word, next_word)
        if trigram_key in self.stats["trigram"]:
            return self.stats["trigram"][trigram_key]

        # 2. Bigram Lookup (Left Context)
        # Checks if the sequence (prev, curr) has been seen
        bigram_key = (prev_word, curr_word)
        if bigram_key in self.stats["bigram"]:
            return self.stats["bigram"][bigram_key]

        # 3. Unigram Lookup (No Context - Memorized Words)
        # Checks if the word itself has a dominant normalization
        unigram_key = curr_word
        if unigram_key in self.stats["unigram"]:
            return self.stats["unigram"][unigram_key]

        # 4. No match found - Delegate to Neural Model
        return None
