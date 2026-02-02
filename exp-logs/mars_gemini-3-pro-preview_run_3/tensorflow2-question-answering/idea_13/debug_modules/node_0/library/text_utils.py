import os
import pandas as pd
import numpy as np
from collections import Counter
from library.config import Config


class TextProcessor:
    """
    Handles text processing tasks including tokenization and document segmentation.
    """

    def __init__(self):
        # Tags that denote the start of a new section/paragraph in Simplified NQ
        self.split_tags = {
            "<P>",
            "<Table>",
            "<Tr>",
            "<Ul>",
            "<Ol>",
            "<Dl>",
            "<H1>",
            "<H2>",
            "<H3>",
            "<H4>",
            "<H5>",
            "<H6>",
            "<Li>",
            "<Dd>",
            "<Dt>",
        }

    def tokenize(self, text):
        """
        Splits text on whitespace.

        Args:
            text (str): Input string.

        Returns:
            list: List of tokens.
        """
        if not text:
            return []
        return text.split()

    def segment_document(self, document_text):
        """
        Segments the document text into candidate paragraphs based on top-level HTML tags.

        Args:
            document_text (str): The raw document text containing HTML tags.

        Returns:
            list: A list of strings, where each string is a candidate paragraph.
        """
        tokens = self.tokenize(document_text)
        candidates = []
        current_tokens = []

        for token in tokens:
            # If the token is a tag that starts a new block, save current buffer and start new
            if token in self.split_tags:
                if current_tokens:
                    candidates.append(" ".join(current_tokens))
                    current_tokens = []
                current_tokens.append(token)
            else:
                current_tokens.append(token)

        # Append any remaining tokens
        if current_tokens:
            candidates.append(" ".join(current_tokens))

        return candidates


class Vocabulary:
    """
    Manages token-to-index mapping and the associated embedding matrix.
    """

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_size = 0
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

    def build(self, texts, max_vocab_size=Config.VOCAB_SIZE):
        """
        Builds the vocabulary from a list of text strings.

        Args:
            texts (iterable): List of strings (questions and document contents).
            max_vocab_size (int): Maximum number of tokens to keep.
        """
        print("Building vocabulary from corpus...")
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        # Initialize with special tokens
        self.token_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_token = {0: self.pad_token, 1: self.unk_token}

        # Add most common tokens up to max_vocab_size
        # Subtract 2 for the special tokens already added
        num_words_to_keep = max_vocab_size - 2
        most_common = counter.most_common(num_words_to_keep)

        for i, (token, _) in enumerate(most_common):
            idx = i + 2
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        self.vocab_size = len(self.token_to_idx)
        print(f"Vocabulary built. Size: {self.vocab_size}")

    def transform(self, text, max_len=None):
        """
        Converts a text string into a list of integer indices.

        Args:
            text (str): Input text.
            max_len (int, optional): If provided, truncates or pads the sequence to this length.

        Returns:
            list: List of integer indices.
        """
        tokens = text.split()
        unk_idx = self.token_to_idx[self.unk_token]
        indices = [self.token_to_idx.get(token, unk_idx) for token in tokens]

        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                pad_idx = self.token_to_idx[self.pad_token]
                indices += [pad_idx] * (max_len - len(indices))

        return indices

    def save(self):
        """
        Saves the vocabulary mapping to a Parquet file.
        """
        os.makedirs(os.path.dirname(Config.VOCAB_PATH), exist_ok=True)

        data = [
            {"token": token, "index": idx} for token, idx in self.token_to_idx.items()
        ]
        df = pd.DataFrame(data)
        df.to_parquet(Config.VOCAB_PATH, index=False)
        print(f"Vocabulary saved to {Config.VOCAB_PATH}")

    def load(self):
        """
        Loads the vocabulary mapping from a Parquet file.
        """
        if not os.path.exists(Config.VOCAB_PATH):
            raise FileNotFoundError(f"Vocabulary file not found at {Config.VOCAB_PATH}")

        df = pd.read_parquet(Config.VOCAB_PATH)
        self.token_to_idx = dict(zip(df["token"], df["index"]))
        self.idx_to_token = dict(zip(df["index"], df["token"]))
        self.vocab_size = len(self.token_to_idx)
        print(f"Vocabulary loaded from {Config.VOCAB_PATH} ({self.vocab_size} tokens)")

    def create_embedding_matrix(self, load_cached_data=False):
        """
        Creates or loads the embedding matrix.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            numpy.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
        """
        # 1. Try to load cached data if requested
        if load_cached_data and os.path.exists(Config.EMBEDDING_MATRIX_PATH):
            try:
                matrix = np.load(Config.EMBEDDING_MATRIX_PATH)
                if matrix.shape == (self.vocab_size, Config.EMBEDDING_DIM):
                    print(
                        f"Loaded embedding matrix from {Config.EMBEDDING_MATRIX_PATH}"
                    )
                    return matrix
                else:
                    print(
                        f"Cached embedding matrix shape {matrix.shape} does not match vocab size ({self.vocab_size}, {Config.EMBEDDING_DIM}). Recreating."
                    )
            except Exception as e:
                print(f"Failed to load cached embedding matrix: {e}. Recreating.")

        # 2. Compute/Create data from scratch
        print("Initializing new embedding matrix...")
        # Since external GloVe files are not available in the read-only input directory,
        # we initialize a random matrix. In a real scenario with internet, we would download GloVe here.
        # Using uniform distribution for initialization
        matrix = np.random.uniform(
            -0.1, 0.1, (self.vocab_size, Config.EMBEDDING_DIM)
        ).astype(np.float32)

        # Ensure padding token is zero vector
        if self.pad_token in self.token_to_idx:
            pad_idx = self.token_to_idx[self.pad_token]
            matrix[pad_idx] = np.zeros(Config.EMBEDDING_DIM)

        # 3. Save result to cache
        os.makedirs(os.path.dirname(Config.EMBEDDING_MATRIX_PATH), exist_ok=True)
        np.save(Config.EMBEDDING_MATRIX_PATH, matrix)
        print(f"Embedding matrix saved to {Config.EMBEDDING_MATRIX_PATH}")

        return matrix


def get_vocab_and_matrix(texts=None, load_cached_data=True):
    """
    Orchestrates the retrieval of the Vocabulary and Embedding Matrix.
    Strictly follows caching logic: Try load -> If fail, Compute & Save.

    Args:
        texts (list, optional): List of texts to build vocab from if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (Vocabulary object, numpy embedding matrix)
    """
    vocab = Vocabulary()
    vocab_loaded = False

    # 1. Try to load vocab
    if load_cached_data:
        try:
            vocab.load()
            vocab_loaded = True
        except FileNotFoundError:
            print("Cached vocabulary not found.")

    # 2. If loading failed or not requested, build from scratch
    if not vocab_loaded:
        if texts is None:
            raise ValueError(
                "Texts must be provided to build vocabulary when cache is unavailable."
            )
        vocab.build(texts)
        vocab.save()

    # 3. Get embedding matrix (handles its own caching logic internally based on vocab)
    embedding_matrix = vocab.create_embedding_matrix(load_cached_data=load_cached_data)

    return vocab, embedding_matrix
