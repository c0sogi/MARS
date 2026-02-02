import os
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Iterable, Optional
from library.config import Config


class TextUtils:
    """
    Utilities for text processing, vocabulary building, and embedding loading.
    Designed for the Siamese Gated Convolutional Ranker.
    """

    # Special tokens constants
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_INDEX = 0
    UNK_INDEX = 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Splits text based on whitespace and punctuation.
        Preserves punctuation as separate tokens and converts text to lowercase.

        Args:
            text: Input string.

        Returns:
            List of string tokens.
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Regex to match alphanumeric sequences or single non-alphanumeric non-whitespace characters
        # This effectively splits punctuation from words
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return tokens

    @staticmethod
    def build_vocab(
        texts: Iterable[str], min_freq: int = 2, load_cached_data: bool = True
    ) -> Dict[str, int]:
        """
        Builds a vocabulary mapping from tokens to integer indices.
        Implements caching using .npy files.

        Args:
            texts: Iterable of text strings to build vocabulary from.
            min_freq: Minimum frequency for a token to be included.
            load_cached_data: Whether to load from cache if available.

        Returns:
            Dictionary mapping tokens to integer indices.
        """
        vocab_path = Config.VOCAB_PATH

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}...")
            try:
                # Load the array of words.
                # We expect a numpy array of strings, which is safe to load without allow_pickle=True
                # if saved correctly as unicode/string dtype.
                vocab_list = np.load(vocab_path)

                # Reconstruct dictionary from the ordered list
                vocab = {word: idx for idx, word in enumerate(vocab_list)}
                print(f"Vocabulary loaded. Size: {len(vocab)}")
                return vocab
            except Exception as e:
                print(f"Failed to load vocabulary cache: {e}. Rebuilding from scratch.")

        # 2. Compute from scratch
        print("Building vocabulary from scratch...")
        counter = Counter()

        # Iterate over texts and update counter
        for i, text in enumerate(texts):
            tokens = TextUtils.tokenize(text)
            counter.update(tokens)
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} texts for vocabulary...")

        # Initialize vocab list with special tokens
        vocab_list = [TextUtils.PAD_TOKEN, TextUtils.UNK_TOKEN]

        # Add words meeting frequency threshold
        # Sorting ensures deterministic behavior
        sorted_words = sorted([w for w, c in counter.items() if c >= min_freq])
        vocab_list.extend(sorted_words)

        # Create dictionary
        vocab = {word: idx for idx, word in enumerate(vocab_list)}

        # 3. Save to cache
        print(f"Saving vocabulary to {vocab_path}...")
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        # Save as a numpy array of strings
        np.save(vocab_path, np.array(vocab_list))

        print(f"Vocabulary built. Size: {len(vocab)}")
        return vocab

    @staticmethod
    def load_glove_embeddings(
        vocab: Dict[str, int],
        glove_path: Optional[str],
        embedding_dim: int = 100,
        load_cached_data: bool = True,
    ) -> np.ndarray:
        """
        Loads pre-trained GloVe embeddings for the vocabulary.
        Implements caching using .npy files.

        Args:
            vocab: Dictionary mapping tokens to indices.
            glove_path: Path to the GloVe text file (e.g., 'glove.6B.100d.txt').
            embedding_dim: Dimension of the embeddings.
            load_cached_data: Whether to load from cache if available.

        Returns:
            Numpy array of shape (vocab_size, embedding_dim).
        """
        cache_path = Config.EMBEDDING_MATRIX_PATH
        vocab_size = len(vocab)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading embedding matrix from {cache_path}...")
            try:
                embedding_matrix = np.load(cache_path)
                if embedding_matrix.shape == (vocab_size, embedding_dim):
                    print("Embedding matrix loaded successfully.")
                    return embedding_matrix
                else:
                    print(
                        f"Cached embedding shape {embedding_matrix.shape} mismatch with vocab size {vocab_size}. Recomputing."
                    )
            except Exception as e:
                print(f"Failed to load embedding cache: {e}. Recomputing.")

        # 2. Compute from scratch
        print("Creating embedding matrix...")

        # Initialize with random normal distribution
        # Scale by 1/sqrt(dim) to keep variance reasonable
        scale = 1.0 / np.sqrt(embedding_dim)
        embedding_matrix = np.random.normal(
            loc=0.0, scale=scale, size=(vocab_size, embedding_dim)
        ).astype(np.float32)

        # Explicitly set PAD token to zeros
        if TextUtils.PAD_TOKEN in vocab:
            embedding_matrix[vocab[TextUtils.PAD_TOKEN]] = np.zeros(
                embedding_dim, dtype=np.float32
            )

        # Parse GloVe file if provided and exists
        if glove_path and os.path.exists(glove_path):
            print(f"Parsing GloVe file: {glove_path}")
            hits = 0
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in vocab:
                        try:
                            # Parse vector
                            vector = np.asarray(values[1:], dtype="float32")
                            if len(vector) == embedding_dim:
                                embedding_matrix[vocab[word]] = vector
                                hits += 1
                        except ValueError:
                            continue
            print(f"Loaded {hits} vectors from GloVe. Coverage: {hits/vocab_size:.2%}")
        else:
            print(
                f"GloVe file not found at '{glove_path}'. Using random initialization for all tokens."
            )

        # 3. Save to cache
        print(f"Saving embedding matrix to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embedding_matrix)

        return embedding_matrix

    @staticmethod
    def text_to_indices(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
        """
        Converts text to a list of indices based on the vocabulary.
        Truncates to max_len or pads with PAD_INDEX to max_len.

        Args:
            text: Input text string.
            vocab: Vocabulary dictionary.
            max_len: Maximum sequence length.

        Returns:
            List of integer indices.
        """
        tokens = TextUtils.tokenize(text)

        # Map to indices, using UNK_INDEX for unknown words
        indices = [vocab.get(token, TextUtils.UNK_INDEX) for token in tokens]

        # Truncate if too long
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad if too short
        if len(indices) < max_len:
            padding = [TextUtils.PAD_INDEX] * (max_len - len(indices))
            indices.extend(padding)

        return indices
