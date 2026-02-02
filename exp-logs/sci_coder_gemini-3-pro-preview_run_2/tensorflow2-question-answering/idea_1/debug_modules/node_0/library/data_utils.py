import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from library.config import Config


class Tokenizer:
    """
    Simple whitespace-based tokenizer that maps tokens to indices.
    Handles vocabulary creation, saving/loading, and sequence conversion.
    """

    def __init__(self):
        self.word2idx = {}
        self.index2word = {}
        self.vocab_size = 0

    def fit(self, token_counts, max_vocab_size=None):
        """
        Fits the tokenizer on token counts.
        Reserves indices 0 for PAD and 1 for UNK.
        """
        # Always reserve 0 for PAD and 1 for UNK
        self.word2idx = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
        self.index2word = {0: Config.PAD_TOKEN, 1: Config.UNK_TOKEN}

        # Sort tokens by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        # Apply vocabulary size limit
        if max_vocab_size:
            # Subtract 2 to account for PAD and UNK
            limit = max_vocab_size - 2
            sorted_tokens = sorted_tokens[:limit]

        for idx, (token, _) in enumerate(sorted_tokens, start=2):
            self.word2idx[token] = idx
            self.index2word[idx] = token

        self.vocab_size = len(self.word2idx)

    def text_to_sequence(self, text_tokens, max_len=None):
        """
        Converts a list of string tokens to a list of integer indices.
        Applies truncation or padding if max_len is provided.
        """
        seq = [
            self.word2idx.get(t, self.word2idx[Config.UNK_TOKEN]) for t in text_tokens
        ]

        if max_len:
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [self.word2idx[Config.PAD_TOKEN]] * (max_len - len(seq))
        return seq

    def save(self, path):
        """Saves the vocabulary mapping to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        """Loads the vocabulary mapping from a JSON file."""
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.index2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)


def extract_candidate_text(doc_tokens, start_token, end_token):
    """
    Extracts text from document tokens based on start and end indices.

    Args:
        doc_tokens (list): List of strings (tokens) from the document.
        start_token (int): Starting index.
        end_token (int): Ending index.

    Returns:
        str: The joined candidate text.
    """
    if start_token < 0 or end_token > len(doc_tokens) or start_token >= end_token:
        return ""
    return " ".join(doc_tokens[start_token:end_token])


def build_tokenizer(
    load_cached_data=True, data_path=Config.TRAIN_DATA_PATH, sample_size=None
):
    """
    Builds or loads a tokenizer with caching.

    Args:
        load_cached_data (bool): Whether to try loading from cache.
        data_path (str): Path to the training data JSONL file.
        sample_size (int, optional): Number of samples to use for building vocab.

    Returns:
        Tokenizer: The fitted tokenizer instance.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(Config.VOCAB_CACHE_PATH), exist_ok=True)

    tokenizer = Tokenizer()

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(Config.VOCAB_CACHE_PATH):
        print(f"Loading tokenizer from {Config.VOCAB_CACHE_PATH}...")
        try:
            tokenizer.load(Config.VOCAB_CACHE_PATH)
            return tokenizer
        except Exception as e:
            print(f"Failed to load tokenizer: {e}. Rebuilding from scratch...")

    # 2. Compute from scratch
    print("Building tokenizer from scratch...")
    token_counts = Counter()

    # Determine sample limit (use Config.DEBUG logic if sample_size not explicit)
    limit = (
        sample_size
        if sample_size is not None
        else (Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None)
    )

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                entry = json.loads(line)

                # Add question tokens
                q_text = entry.get("question_text", "")
                q_tokens = q_text.split()
                token_counts.update(q_tokens)

                # Add document tokens
                doc_text = entry.get("document_text", "")
                doc_tokens = doc_text.split()
                token_counts.update(doc_tokens)

    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Returning empty tokenizer.")
        return tokenizer

    tokenizer.fit(token_counts, max_vocab_size=Config.VOCAB_SIZE)

    # 3. Save to cache
    print(f"Saving tokenizer to {Config.VOCAB_CACHE_PATH}...")
    tokenizer.save(Config.VOCAB_CACHE_PATH)

    return tokenizer


def build_idf_weights(
    tokenizer, load_cached_data=True, data_path=Config.TRAIN_DATA_PATH, sample_size=None
):
    """
    Computes or loads IDF weights for the vocabulary.
    IDF(t) = log(N / (df(t) + 1)) + 1

    Args:
        tokenizer (Tokenizer): Fitted tokenizer.
        load_cached_data (bool): Whether to try loading from cache.
        data_path (str): Path to training data.
        sample_size (int, optional): Limit on samples.

    Returns:
        np.array: Array of IDF weights of shape (vocab_size,).
    """
    os.makedirs(os.path.dirname(Config.IDF_CACHE_PATH), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(Config.IDF_CACHE_PATH):
        print(f"Loading IDF weights from {Config.IDF_CACHE_PATH}...")
        try:
            idf_weights = np.load(Config.IDF_CACHE_PATH)
            if len(idf_weights) == tokenizer.vocab_size:
                return idf_weights
            else:
                print(
                    f"Cached IDF dimension {len(idf_weights)} != Vocab size {tokenizer.vocab_size}. Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load IDF weights: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing IDF weights...")
    doc_freqs = defaultdict(int)
    total_docs = 0

    limit = (
        sample_size
        if sample_size is not None
        else (Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None)
    )

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                entry = json.loads(line)
                total_docs += 1

                # Combine question and document to form the context for DF calculation
                text = (
                    entry.get("document_text", "")
                    + " "
                    + entry.get("question_text", "")
                )
                tokens = set(text.split())  # Unique tokens in this document

                for token in tokens:
                    if token in tokenizer.word2idx:
                        idx = tokenizer.word2idx[token]
                        doc_freqs[idx] += 1

    except FileNotFoundError:
        print("Data file not found. Returning default weights.")
        return np.ones(tokenizer.vocab_size)

    # Calculate IDF
    idf_weights = np.zeros(tokenizer.vocab_size)

    for idx in range(tokenizer.vocab_size):
        df = doc_freqs.get(idx, 0)
        # Smooth IDF calculation
        idf_weights[idx] = np.log((total_docs + 1) / (df + 1)) + 1

    # 3. Save to cache
    print(f"Saving IDF weights to {Config.IDF_CACHE_PATH}...")
    np.save(Config.IDF_CACHE_PATH, idf_weights)

    return idf_weights
