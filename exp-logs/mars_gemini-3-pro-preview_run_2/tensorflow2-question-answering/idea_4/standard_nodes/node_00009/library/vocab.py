import os
import json
import collections
import numpy as np
import pandas as pd
from library.config import Config


class Vocabulary:
    """
    Manages the mapping between tokens and their integer indices.
    """

    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = []

        # Add special tokens immediately
        self.add_token(Config.PAD_TOKEN)  # Index 0
        self.add_token(Config.UNK_TOKEN)  # Index 1

    def add_token(self, token):
        """Adds a token to the vocabulary if it doesn't exist."""
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def lookup_token(self, token):
        """Returns the index of the token, or the UNK index if not found."""
        return self.token_to_index.get(token, self.token_to_index[Config.UNK_TOKEN])

    def lookup_index(self, index):
        """Returns the token for a given index."""
        if 0 <= index < len(self.index_to_token):
            return self.index_to_token[index]
        return Config.UNK_TOKEN

    def __len__(self):
        return len(self.index_to_token)

    def save(self, path):
        """Saves the vocabulary list to a .npy file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.index_to_token))
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """Loads the vocabulary list from a .npy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        vocab_array = np.load(path)
        self.index_to_token = vocab_array.tolist()
        self.token_to_index = {token: i for i, token in enumerate(self.index_to_token)}
        print(f"Vocabulary loaded from {path}. Size: {len(self)}")


def build_vocab(data_path, load_cached_data=True):
    """
    Builds the vocabulary from the training corpus or loads it from cache.

    Args:
        data_path (str): Path to the training JSONL file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Vocabulary: The constructed or loaded Vocabulary object.
    """
    vocab = Vocabulary()
    cache_path = Config.VOCAB_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            vocab.load(cache_path)
            return vocab
        except Exception as e:
            print(f"Failed to load cached vocabulary: {e}. Rebuilding...")

    # 2. Build from scratch
    print(f"Building vocabulary from {data_path}...")

    # Counter for frequency analysis
    token_counts = collections.Counter()

    # Determine sample size for debugging
    limit = Config.DEBUG_SAMPLE_SIZE

    try:
        # Read file line by line to be memory efficient
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break

                entry = json.loads(line)

                # Tokenize Question
                q_text = entry.get("question_text", "")
                token_counts.update(q_text.split())

                # Tokenize Document
                doc_text = entry.get("document_text", "")
                token_counts.update(doc_text.split())

    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found.")
        # Return basic vocab with just special tokens if file missing
        return vocab

    # Select top N most frequent words, excluding special tokens (already added)
    # Capacity for new words = VOCAB_SIZE - current_size (2)
    capacity = Config.VOCAB_SIZE - len(vocab)
    most_common = token_counts.most_common(capacity)

    for token, _ in most_common:
        vocab.add_token(token)

    print(f"Vocabulary built. Size: {len(vocab)}")

    # 3. Save to cache
    vocab.save(cache_path)

    return vocab


def build_embedding_matrix(vocab, glove_path=None, load_cached_data=True):
    """
    Creates the embedding matrix. Loads from cache if available, otherwise
    loads GloVe vectors or initializes randomly.

    Args:
        vocab (Vocabulary): The vocabulary object.
        glove_path (str, optional): Path to the GloVe text file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = Config.EMBEDDING_MATRIX_PATH
    vocab_size = len(vocab)
    embed_dim = Config.EMBEDDING_DIM

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            embedding_matrix = np.load(cache_path)
            # Verify shape matches current vocab and config
            if embedding_matrix.shape == (vocab_size, embed_dim):
                print(f"Embedding matrix loaded from {cache_path}")
                return embedding_matrix
            else:
                print(
                    f"Cached embedding shape {embedding_matrix.shape} mismatch with expected {(vocab_size, embed_dim)}. Rebuilding..."
                )
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Building embedding matrix...")

    # Initialize randomly (normal distribution)
    # Scale by 1/sqrt(dim) to keep variance reasonable
    scale = 1.0 / np.sqrt(embed_dim)
    embedding_matrix = np.random.normal(
        loc=0.0, scale=scale, size=(vocab_size, embed_dim)
    )

    # Zero out padding token
    pad_idx = vocab.lookup_token(Config.PAD_TOKEN)
    embedding_matrix[pad_idx] = np.zeros(embed_dim)

    # Load GloVe if provided and exists
    if glove_path and os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from {glove_path}...")
        hits = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]

                # Check if word is in our vocab
                # Note: This lookup is O(1)
                if word in vocab.token_to_index:
                    idx = vocab.token_to_index[word]
                    try:
                        vector = np.asarray(values[1:], dtype="float32")
                        if vector.shape[0] == embed_dim:
                            embedding_matrix[idx] = vector
                            hits += 1
                    except ValueError:
                        continue

        print(f"Loaded {hits} vectors from GloVe. Coverage: {hits/vocab_size:.2%}")
    else:
        print("GloVe path not provided or file not found. Using random initialization.")

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embedding_matrix)
    print(f"Embedding matrix saved to {cache_path}")

    return embedding_matrix
