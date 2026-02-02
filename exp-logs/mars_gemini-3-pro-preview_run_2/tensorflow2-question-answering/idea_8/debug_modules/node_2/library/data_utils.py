import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from library.config import config


def tokenize(text):
    """
    Splits text into tokens based on whitespace.

    Args:
        text (str): Input text.

    Returns:
        list: List of string tokens.
    """
    if not text:
        return []
    return text.split()


def build_vocab(load_cached_data=True):
    """
    Builds or loads a vocabulary from the training data.

    The vocabulary is saved as a numpy array of strings (the keys) to avoid
    pickling python dictionaries directly. The index is implicit based on position.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: A mapping from token (str) to index (int).
    """
    vocab_path = config.VOCAB_CACHE_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(vocab_path):
        try:
            # Load the array of words
            words = np.load(vocab_path)
            # Reconstruct the dictionary: word -> index
            vocab = {word: idx for idx, word in enumerate(words)}
            print(f"Loaded vocabulary from {vocab_path}. Size: {len(vocab)}")
            return vocab
        except Exception as e:
            print(f"Failed to load vocabulary cache: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Building vocabulary from training data...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    token_counter = Counter()

    # Read training data in chunks to be memory efficient
    # We only need document_text and question_text
    try:
        chunks = pd.read_json(config.TRAIN_DATA_PATH, lines=True, chunksize=10000)
        for chunk in chunks:
            # Process questions
            for q_text in chunk["question_text"]:
                token_counter.update(tokenize(q_text))

            # Process documents (these are long, so this takes time)
            # We might limit this if dataset is huge, but for NQ simplified we process all
            for doc_text in chunk["document_text"]:
                token_counter.update(tokenize(doc_text))

    except ValueError as e:
        print(f"Error reading training data: {e}")
        # Fallback for empty or missing file scenarios (mainly for testing pipeline flow)
        pass

    # Select top N words
    # We reserve 0 for PAD and 1 for UNK
    # So we take VOCAB_SIZE - 2 most common words
    most_common = token_counter.most_common(config.VOCAB_SIZE - 2)

    # Create the ordered list of words
    # Index 0: <PAD>
    # Index 1: <UNK>
    words_list = [config.PAD_TOKEN, config.UNK_TOKEN] + [
        token for token, count in most_common
    ]

    # Convert to numpy array for saving
    words_array = np.array(words_list)

    # Save to cache
    np.save(vocab_path, words_array)
    print(f"Saved vocabulary to {vocab_path}. Size: {len(words_array)}")

    # Create dictionary
    vocab = {word: idx for idx, word in enumerate(words_list)}

    return vocab


def load_embeddings(vocab, load_cached_data=True):
    """
    Loads or initializes the embedding matrix.

    Args:
        vocab (dict): Vocabulary mapping token to index.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    emb_path = config.EMBEDDING_MATRIX_CACHE_PATH
    vocab_size = len(vocab)
    embedding_dim = config.EMBEDDING_DIM

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(emb_path):
        try:
            embedding_matrix = np.load(emb_path)
            # Check if dimensions match current config
            if embedding_matrix.shape == (vocab_size, embedding_dim):
                print(
                    f"Loaded embedding matrix from {emb_path}. Shape: {embedding_matrix.shape}"
                )
                return embedding_matrix
            else:
                print(
                    f"Cached embedding matrix shape {embedding_matrix.shape} mismatch with config {(vocab_size, embedding_dim)}. Rebuilding..."
                )
        except Exception as e:
            print(f"Failed to load embedding cache: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Initializing embedding matrix...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    # Initialize random embeddings
    # We use a normal distribution scaled by 1/sqrt(dim) for better convergence
    scale = 1.0 / np.sqrt(embedding_dim)
    embedding_matrix = np.random.normal(
        loc=0.0, scale=scale, size=(vocab_size, embedding_dim)
    )

    # Set PAD token embedding to zeros
    pad_idx = vocab.get(config.PAD_TOKEN, 0)
    embedding_matrix[pad_idx] = np.zeros(embedding_dim)

    # If using pretrained embeddings (Placeholder logic)
    if config.USE_PRETRAINED_EMBEDDINGS:
        print(
            "Note: Pretrained embedding loading is enabled in config, but no source path is defined. Using random initialization."
        )
        # Logic to load GloVe or FastText would go here
        # iterating through the file and updating embedding_matrix[vocab[word]]

    # Save to cache
    np.save(emb_path, embedding_matrix)
    print(f"Saved embedding matrix to {emb_path}.")

    return embedding_matrix


def text_to_indices(text, vocab):
    """
    Converts a text string to a list of integer indices.

    Args:
        text (str): Input text.
        vocab (dict): Vocabulary mapping.

    Returns:
        list: List of token indices.
    """
    tokens = tokenize(text)
    unk_idx = vocab.get(config.UNK_TOKEN, 1)
    return [vocab.get(t, unk_idx) for t in tokens]


def pad_sequence(sequence, max_len):
    """
    Pads or truncates a sequence of indices to a fixed length.

    Args:
        sequence (list): List of integer indices.
        max_len (int): Target length.

    Returns:
        np.ndarray: Padded sequence of shape (max_len,).
    """
    seq_len = len(sequence)
    pad_idx = 0  # Assuming PAD is always 0 based on build_vocab

    if seq_len >= max_len:
        return np.array(sequence[:max_len], dtype=np.int64)
    else:
        # Create array filled with pad_idx
        padded = np.full(max_len, pad_idx, dtype=np.int64)
        # Fill beginning with sequence
        padded[:seq_len] = sequence
        return padded
