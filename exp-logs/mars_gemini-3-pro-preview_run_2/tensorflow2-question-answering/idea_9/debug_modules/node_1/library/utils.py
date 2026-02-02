import os
import numpy as np
from collections import Counter
from typing import List, Dict, Iterable, Optional
from library.config import Config


def tokenize(text: str) -> List[str]:
    """
    Splits the input text into a list of tokens using whitespace splitting.

    Args:
        text: The input string.

    Returns:
        A list of string tokens.
    """
    if not text:
        return []
    return text.split()


def build_vocab(
    texts: Optional[Iterable[str]] = None, load_cached_data: bool = True
) -> Dict[str, int]:
    """
    Constructs a vocabulary mapping from tokens to indices.

    Implements strict caching logic:
    1. If load_cached_data is True, attempts to load from Config.VOCAB_CACHE_PATH.
    2. If loading fails or load_cached_data is False, computes vocabulary from `texts`.
    3. Saves the computed vocabulary to cache.

    Args:
        texts: An iterable of strings to build the vocabulary from. Can be None if loading from cache.
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        A dictionary mapping token strings to integer indices.
    """
    vocab_path = Config.VOCAB_CACHE_PATH
    vocab_list = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(vocab_path):
        try:
            print(f"Loading vocabulary from {vocab_path}...")
            vocab_list = np.load(vocab_path)
            print(f"Loaded vocabulary with {len(vocab_list)} tokens.")
        except Exception as e:
            print(f"Failed to load vocabulary cache: {e}")
            vocab_list = None

    # 2. Compute from scratch if needed
    if vocab_list is None:
        if texts is None:
            raise ValueError(
                "No cached vocabulary found and no input texts provided to build one."
            )

        print("Building vocabulary from corpus...")
        counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        # Reserve spots for special tokens
        num_special = 3  # PAD, UNK, SEP
        max_vocab = Config.VOCAB_SIZE
        num_tokens_to_keep = max_vocab - num_special

        most_common = counter.most_common(num_tokens_to_keep)

        # Create list: [PAD, UNK, SEP, token1, token2, ...]
        # Indices: 0, 1, 2, 3, 4, ...
        vocab_list = [Config.PAD_TOKEN, Config.UNK_TOKEN, Config.SEP_TOKEN] + [
            token for token, count in most_common
        ]
        vocab_list = np.array(vocab_list)

        # 3. Save to cache
        print(f"Saving vocabulary to {vocab_path}...")
        np.save(vocab_path, vocab_list)

    # Convert list to dict for O(1) lookups
    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
    return vocab_dict


def create_embedding_matrix(
    vocab: Dict[str, int], load_cached_data: bool = True
) -> np.ndarray:
    """
    Creates an embedding matrix for the given vocabulary.

    Implements strict caching logic:
    1. If load_cached_data is True, attempts to load from Config.EMBEDDING_MATRIX_CACHE_PATH.
    2. If loading fails or load_cached_data is False, initializes a new random matrix.
    3. Saves the matrix to cache.

    Args:
        vocab: Dictionary mapping tokens to indices.
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        A numpy array of shape (vocab_size, embedding_dim).
    """
    emb_path = Config.EMBEDDING_MATRIX_CACHE_PATH
    matrix = None

    expected_shape = (len(vocab), Config.EMBEDDING_DIM)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(emb_path):
        try:
            print(f"Loading embedding matrix from {emb_path}...")
            matrix = np.load(emb_path)
            if matrix.shape != expected_shape:
                print(
                    f"Cached matrix shape {matrix.shape} does not match vocab size {expected_shape}. Recomputing."
                )
                matrix = None
            else:
                print("Loaded embedding matrix successfully.")
        except Exception as e:
            print(f"Failed to load embedding matrix cache: {e}")
            matrix = None

    # 2. Compute (Initialize) from scratch if needed
    if matrix is None:
        print(f"Initializing random embedding matrix with shape {expected_shape}...")
        # Initialize with random normal distribution
        # In a full implementation, this is where we would load GloVe and map to vocab
        matrix = np.random.normal(scale=0.1, size=expected_shape).astype(np.float32)

        # Explicitly set PAD token vector to zero if it exists in vocab
        if Config.PAD_TOKEN in vocab:
            pad_idx = vocab[Config.PAD_TOKEN]
            matrix[pad_idx] = np.zeros(Config.EMBEDDING_DIM)

        # 3. Save to cache
        print(f"Saving embedding matrix to {emb_path}...")
        np.save(emb_path, matrix)

    return matrix
