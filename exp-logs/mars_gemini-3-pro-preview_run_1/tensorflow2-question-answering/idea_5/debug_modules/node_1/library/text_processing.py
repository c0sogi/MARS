import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import torch
from library.config import Config


def tokenize(text):
    """
    Simple whitespace tokenizer.
    Converts text to lowercase and splits by whitespace.

    Args:
        text (str): Input text string.

    Returns:
        list: List of tokens.
    """
    if not text:
        return []
    return text.lower().split()


def build_vocab(load_cached_data=True):
    """
    Builds a vocabulary from the training data.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: Mapping from token to integer index.
    """
    vocab_path = Config.VOCAB_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(vocab_path):
        print(f"[TextProcessing] Loading vocab from {vocab_path}")
        try:
            df = pd.read_parquet(vocab_path)
            # Convert to dict: token -> index
            vocab = dict(zip(df["token"], df["index"]))
            return vocab
        except Exception as e:
            print(f"[TextProcessing] Failed to load cached vocab: {e}. Rebuilding...")

    # 2. Rebuild from scratch
    print("[TextProcessing] Building vocabulary from training data...")

    # Load metadata to access training samples
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_META_PATH}"
        )

    meta_df = pd.read_parquet(Config.TRAIN_META_PATH)

    # Handle Debug mode
    if Config.DEBUG:
        meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(
            f"[TextProcessing] DEBUG mode: Using {len(meta_df)} samples for vocab build."
        )

    jsonl_path = Config.TRAIN_DATA_PATH
    counter = Counter()

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Training data file not found at {jsonl_path}")

    with open(jsonl_path, "rb") as f:
        for _, row in meta_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Tokenize question
                q_tokens = tokenize(data.get("question_text", ""))
                counter.update(q_tokens)

                # Tokenize document text
                doc_tokens = tokenize(data.get("document_text", ""))
                counter.update(doc_tokens)

            except json.JSONDecodeError:
                continue

    # Select top N words
    # Reserve 0 for PAD, 1 for UNK
    # Actual words start at 2
    # We subtract 2 from MAX_VOCAB_SIZE to account for PAD and UNK
    most_common = counter.most_common(Config.MAX_VOCAB_SIZE - 2)

    vocab = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
    for token, _ in most_common:
        vocab[token] = len(vocab)

    print(f"[TextProcessing] Vocab built. Size: {len(vocab)}")

    # 3. Save to cache
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    # Create DataFrame for Parquet storage
    vocab_list = [{"token": k, "index": v} for k, v in vocab.items()]
    vocab_df = pd.DataFrame(vocab_list)
    vocab_df.to_parquet(vocab_path, index=False)
    print(f"[TextProcessing] Vocab saved to {vocab_path}")

    return vocab


def create_embedding_matrix(vocab, load_cached_data=True):
    """
    Creates an embedding matrix corresponding to the vocabulary.
    Initializes randomly since no specific pre-trained file is provided in input.

    Args:
        vocab (dict): Token to index mapping.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    emb_path = Config.EMBEDDING_MATRIX_PATH
    vocab_size = len(vocab)
    emb_dim = Config.EMBEDDING_DIM

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(emb_path):
        print(f"[TextProcessing] Loading embedding matrix from {emb_path}")
        try:
            matrix = np.load(emb_path)
            if matrix.shape == (vocab_size, emb_dim):
                return matrix
            else:
                print(
                    f"[TextProcessing] Cached matrix shape {matrix.shape} mismatch with vocab {vocab_size}. Rebuilding..."
                )
        except Exception as e:
            print(
                f"[TextProcessing] Failed to load cached embeddings: {e}. Rebuilding..."
            )

    # 2. Build/Initialize
    print(
        f"[TextProcessing] Initializing embedding matrix ({vocab_size}, {emb_dim})..."
    )

    # Initialize with random normal distribution
    # Scale by 1/sqrt(dim) for better convergence stability
    scale = 1.0 / np.sqrt(emb_dim)
    matrix = np.random.normal(loc=0.0, scale=scale, size=(vocab_size, emb_dim))

    # Explicitly set PAD to zeros
    if Config.PAD_TOKEN in vocab:
        pad_idx = vocab[Config.PAD_TOKEN]
        matrix[pad_idx] = np.zeros(emb_dim)

    # 3. Save to cache
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.save(emb_path, matrix)
    print(f"[TextProcessing] Embedding matrix saved to {emb_path}")

    return matrix


def text_to_indices(text, vocab, max_len):
    """
    Converts a text string into a list of integer indices based on the vocab.
    Truncates or pads to max_len.

    Args:
        text (str): Input text.
        vocab (dict): Vocab mapping.
        max_len (int): Maximum sequence length.

    Returns:
        list: List of integers.
    """
    tokens = tokenize(text)
    unk_idx = vocab.get(Config.UNK_TOKEN, 1)
    pad_idx = vocab.get(Config.PAD_TOKEN, 0)

    indices = [vocab.get(t, unk_idx) for t in tokens]

    # Truncate
    if len(indices) > max_len:
        indices = indices[:max_len]

    # Pad
    if len(indices) < max_len:
        indices += [pad_idx] * (max_len - len(indices))

    return indices
