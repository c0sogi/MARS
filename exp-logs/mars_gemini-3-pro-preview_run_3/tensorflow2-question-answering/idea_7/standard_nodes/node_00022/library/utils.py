import os
import json
import pandas as pd
import numpy as np
from collections import Counter

# Configuration
CACHE_DIR = "./working/idea_7/"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"

# Hyperparameters (Defaults)
DEFAULT_VOCAB_SIZE = 50000
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_MAX_SEQ_LEN = 512
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


def ensure_dir(directory):
    """Ensures the directory exists."""
    os.makedirs(directory, exist_ok=True)


class Tokenizer:
    """
    Custom tokenizer for the Natural Questions dataset.
    Splits on whitespace as per dataset spec and maps tokens to indices.
    """

    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.index_word = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.vocab_count = 2

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.
        Args:
            texts: List of strings (documents or questions).
        """
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = text.split()
                counter.update(tokens)

        # Keep top vocab_size - 2 (for PAD and UNK)
        most_common = counter.most_common(self.vocab_size - 2)

        for word, _ in most_common:
            if word not in self.word_index:
                self.word_index[word] = self.vocab_count
                self.index_word[self.vocab_count] = word
                self.vocab_count += 1

    def texts_to_sequences(self, texts, max_len=None):
        """
        Converts texts to list of indices.
        Args:
            texts: List of strings.
            max_len: Optional integer to truncate/pad sequences.
        Returns:
            List of list of integers.
        """
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                seq = []
            else:
                tokens = text.split()
                seq = [
                    self.word_index.get(t, self.word_index[UNK_TOKEN]) for t in tokens
                ]

            if max_len:
                if len(seq) > max_len:
                    seq = seq[:max_len]
                else:
                    seq = seq + [self.word_index[PAD_TOKEN]] * (max_len - len(seq))
            sequences.append(seq)
        return sequences

    def save(self, cache_path):
        """Saves vocabulary to parquet."""
        data = list(self.word_index.items())
        df = pd.DataFrame(data, columns=["word", "index"])
        ensure_dir(os.path.dirname(cache_path))
        df.to_parquet(cache_path, index=False)
        print(f"Tokenizer vocabulary saved to {cache_path}")

    def load(self, cache_path):
        """Loads vocabulary from parquet."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Vocab file not found at {cache_path}")

        df = pd.read_parquet(cache_path)
        self.word_index = dict(zip(df["word"], df["index"]))
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.vocab_count = len(self.word_index)
        print(
            f"Tokenizer vocabulary loaded from {cache_path}. Size: {self.vocab_count}"
        )


def parse_candidates(document_text, long_answer_candidates):
    """
    Parses document text into discrete paragraph candidates based on token spans.

    Args:
        document_text (str): The raw text of the article.
        long_answer_candidates (list): List of dicts from the dataset JSON
                                       containing 'start_token', 'end_token', 'top_level'.

    Returns:
        list: A list of strings, where each string is a candidate paragraph.
    """
    if not document_text:
        return []

    tokens = document_text.split()
    candidates = []

    for cand in long_answer_candidates:
        # We typically focus on top_level candidates for long answers
        if cand.get("top_level", False):
            start = cand["start_token"]
            end = cand["end_token"]
            # Ensure indices are within bounds
            if start < len(tokens) and end <= len(tokens):
                candidate_tokens = tokens[start:end]
                candidates.append(" ".join(candidate_tokens))

    return candidates


def load_embeddings(
    vocab_word_index,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    embedding_file=None,
    load_cached_data=True,
):
    """
    Loads pre-trained embeddings or initializes random ones.
    Implements caching using .npy files.

    Args:
        vocab_word_index (dict): Dictionary mapping words to indices.
        embedding_dim (int): Dimension of embedding vectors.
        embedding_file (str): Path to GloVe text file (optional).
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = os.path.join(CACHE_DIR, f"embedding_matrix_{embedding_dim}d.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embedding matrix from cache: {cache_path}")
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing.")

    # 2. Compute from scratch
    print("Initializing embedding matrix...")
    vocab_size = len(vocab_word_index)

    # Initialize with random normal distribution
    np.random.seed(42)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

    # Zero out PAD token
    if PAD_TOKEN in vocab_word_index:
        embedding_matrix[vocab_word_index[PAD_TOKEN]] = np.zeros(embedding_dim)

    # If an embedding file is provided and exists, load vectors
    if embedding_file and os.path.exists(embedding_file):
        print(f"Loading vectors from {embedding_file}...")
        hits = 0
        with open(embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab_word_index:
                    try:
                        vector = np.asarray(values[1:], dtype="float32")
                        if len(vector) == embedding_dim:
                            embedding_matrix[vocab_word_index[word]] = vector
                            hits += 1
                    except ValueError:
                        continue
        print(f"Loaded {hits} vectors from file.")
    else:
        print(
            "No external embedding file provided or found. Using random initialization."
        )

    # 3. Save to cache
    ensure_dir(CACHE_DIR)
    np.save(cache_path, embedding_matrix)
    print(f"Saved embedding matrix to {cache_path}")

    return embedding_matrix.astype(np.float32)


def load_jsonl_sample(file_path, offset):
    """
    Reads a single JSON line from a file at a specific byte offset.

    Args:
        file_path (str): Path to the .jsonl file.
        offset (int): Byte offset to seek to.

    Returns:
        dict: Parsed JSON object.
    """
    with open(file_path, "rb") as f:
        f.seek(offset)
        line = f.readline()
        if line:
            return json.loads(line.decode("utf-8"))
    return None


def get_dataset_partitions(metadata_dir=METADATA_DIR):
    """
    Loads the train, validation, and test metadata dataframes.

    Args:
        metadata_dir (str): Directory containing metadata CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_path = os.path.join(metadata_dir, "test_metadata.csv")

    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

    return train_df, val_df, test_df
