import os
import collections
import pandas as pd
import numpy as np
from typing import List, Iterable, Optional, Tuple, Dict
from library.config import (
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    MAX_VOCAB_SIZE,
    MIN_FREQ,
    CACHE_DIR,
)


class Vocabulary:
    """
    A class to handle mapping between tokens (strings) and IDs (integers).
    Supports saving/loading to Parquet to avoid pickle.
    """

    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Initialize with special tokens
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def build_from_corpus(
        self, corpus: Iterable[str], min_freq: int = 1, max_size: Optional[int] = None
    ) -> None:
        """
        Builds the vocabulary from an iterable of tokens.

        Args:
            corpus: Iterable of strings (tokens).
            min_freq: Minimum frequency to include a token.
            max_size: Maximum vocabulary size (excluding special tokens).
        """
        counter = collections.Counter(corpus)

        # Sort by frequency (descending) then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        added_count = 0
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size is not None and added_count >= max_size:
                break

            self._add_token(token)
            added_count += 1

    def build_from_unique_tokens(self, tokens: Iterable[str]) -> None:
        """
        Builds vocabulary from a unique set/list of tokens (no frequency filtering).
        Useful for classes or characters where we want everything.
        """
        # Sort for deterministic ID assignment
        sorted_tokens = sorted(list(set(tokens)))
        for token in sorted_tokens:
            self._add_token(token)

    def __len__(self) -> int:
        return len(self.token2id)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """
        Converts a list of tokens to IDs. Replaces unknown tokens with UNK_TOKEN if available.
        """
        unk_id = self.token2id.get(UNK_TOKEN)
        ids = []
        for token in tokens:
            idx = self.token2id.get(token)
            if idx is None:
                if unk_id is not None:
                    ids.append(unk_id)
                else:
                    # If no UNK token is defined (e.g. for classes), we default to 0
                    # assuming 0 is a safe fallback or padding.
                    ids.append(0)
            else:
                ids.append(idx)
        return ids

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """
        Converts a list of IDs back to tokens.
        """
        return [self.id2token.get(idx, UNK_TOKEN) for idx in indices]

    def save(self, filepath: str) -> None:
        """
        Saves the vocabulary mapping to a parquet file.
        """
        data = [{"token": token, "id": idx} for token, idx in self.token2id.items()]
        df = pd.DataFrame(data)
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)

    def load(self, filepath: str) -> None:
        """
        Loads the vocabulary mapping from a parquet file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")

        df = pd.read_parquet(filepath)
        # Ensure correct types
        df["token"] = df["token"].astype(str)
        df["id"] = df["id"].astype(int)

        self.token2id = dict(zip(df["token"], df["id"]))
        self.id2token = dict(zip(df["id"], df["token"]))


def build_vocabularies(
    df_train: pd.DataFrame, load_cached_data: bool = True
) -> Tuple[Vocabulary, Vocabulary, Vocabulary]:
    """
    Constructs or loads the Word, Character, and Class vocabularies.

    Args:
        df_train: The training dataframe containing 'before', 'after', and 'class' columns.
        load_cached_data: If True, attempts to load from CACHE_DIR.

    Returns:
        (vocab_words, vocab_chars, vocab_classes)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    path_words = os.path.join(CACHE_DIR, "vocab_words.parquet")
    path_chars = os.path.join(CACHE_DIR, "vocab_chars.parquet")
    path_classes = os.path.join(CACHE_DIR, "vocab_classes.parquet")

    # Initialize Vocabularies with appropriate special tokens
    vocab_words = Vocabulary(special_tokens=[PAD_TOKEN, UNK_TOKEN])
    vocab_chars = Vocabulary(
        special_tokens=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    )
    vocab_classes = Vocabulary(special_tokens=[])

    # Check if all exist
    all_exist = (
        os.path.exists(path_words)
        and os.path.exists(path_chars)
        and os.path.exists(path_classes)
    )

    if load_cached_data and all_exist:
        print("Loading vocabularies from cache...")
        vocab_words.load(path_words)
        vocab_chars.load(path_chars)
        vocab_classes.load(path_classes)
    else:
        print("Building vocabularies from source data...")

        # 1. Build Word Vocabulary
        # We use the 'before' column (raw text)
        words = df_train["before"].astype(str).tolist()
        vocab_words.build_from_corpus(words, min_freq=MIN_FREQ, max_size=MAX_VOCAB_SIZE)

        # 2. Build Character Vocabulary
        # We need characters from both 'before' and 'after' for the Seq2Seq model
        unique_chars = set()

        # Process 'before' column
        for text in df_train["before"].astype(str):
            unique_chars.update(text)

        # Process 'after' column
        for text in df_train["after"].astype(str):
            unique_chars.update(text)

        # Build vocab from unique characters
        vocab_chars.build_from_unique_tokens(unique_chars)

        # 3. Build Class Vocabulary
        # Just the unique classes found in the training set
        unique_classes = df_train["class"].astype(str).unique()
        vocab_classes.build_from_unique_tokens(unique_classes)

        # Save to cache
        print(f"Saving vocabularies to {CACHE_DIR}...")
        vocab_words.save(path_words)
        vocab_chars.save(path_chars)
        vocab_classes.save(path_classes)

    print(f"Word Vocab Size: {len(vocab_words)}")
    print(f"Char Vocab Size: {len(vocab_chars)}")
    print(f"Class Vocab Size: {len(vocab_classes)}")

    return vocab_words, vocab_chars, vocab_classes
