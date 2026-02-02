import os
import json
import numpy as np
from collections import Counter
from library.config import Config


class VocabManager:
    """
    Manages vocabulary creation, loading, and text-to-index conversion.
    Handles caching of the vocabulary and embedding matrix.
    """

    def __init__(self, config=Config):
        self.config = config
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.embedding_matrix = None
        self.vocab_size = 0

        # Special tokens
        self.pad_token = config.PAD_TOKEN
        self.unk_token = config.UNK_TOKEN
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, load_cached_data=True):
        """
        Builds the vocabulary from the training data or loads it from cache.

        Args:
            load_cached_data (bool): If True, tries to load from disk first.
        """
        vocab_path = self.config.VOCAB_PATH
        embedding_path = self.config.EMBEDDING_MATRIX_PATH

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(vocab_path) and os.path.exists(embedding_path):
                print(f"Loading vocabulary from {vocab_path}...")
                try:
                    # Load vocabulary list (numpy array of strings)
                    vocab_list = np.load(vocab_path)
                    self._set_vocab_from_list(vocab_list)

                    # Load embedding matrix
                    print(f"Loading embedding matrix from {embedding_path}...")
                    self.embedding_matrix = np.load(embedding_path)

                    print(f"Loaded vocabulary size: {self.vocab_size}")
                    return
                except Exception as e:
                    print(f"Failed to load cached data: {e}. Recomputing...")
            else:
                print(
                    "Cached vocabulary or embeddings not found. Computing from scratch..."
                )
        else:
            print("Force recompute enabled. Computing vocabulary from scratch...")

        # 2. Compute from scratch
        self._compute_vocab_and_embeddings()

        # 3. Save to cache
        self._save_cache()

    def _compute_vocab_and_embeddings(self):
        """
        Reads the training file, counts tokens, builds the vocabulary,
        and initializes the embedding matrix.
        """
        train_path = self.config.TRAIN_DATA_PATH
        counter = Counter()

        print(f"Reading data from {train_path}...")

        limit = self.config.DEBUG_SIZE if self.config.DEBUG else None
        count = 0

        try:
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit and count >= limit:
                        break

                    entry = json.loads(line)

                    # Tokenize document text
                    doc_text = entry.get("document_text", "")
                    doc_tokens = doc_text.split()
                    counter.update(doc_tokens)

                    # Tokenize question text
                    q_text = entry.get("question_text", "")
                    q_tokens = q_text.split()
                    counter.update(q_tokens)

                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} lines for vocabulary...")

        except FileNotFoundError:
            raise FileNotFoundError(f"Training data file not found at {train_path}")

        print(f"Total unique tokens found: {len(counter)}")

        # Select top N tokens
        # Reserve 2 slots for PAD and UNK
        max_vocab = self.config.VOCAB_SIZE - 2
        most_common = counter.most_common(max_vocab)

        # Create vocab list: [PAD, UNK, word1, word2, ...]
        vocab_list = [self.pad_token, self.unk_token] + [
            token for token, freq in most_common
        ]

        self._set_vocab_from_list(vocab_list)
        print(f"Final vocabulary size: {self.vocab_size}")

        # Initialize Embedding Matrix
        # In a real scenario with internet, we would load GloVe here.
        # Since we are restricted, we initialize randomly.
        # We assume embeddings are 'pre-trained' in the sense that they are fixed
        # after this initialization step.
        print("Initializing random embedding matrix...")
        self.embedding_matrix = np.random.normal(
            scale=0.6, size=(self.vocab_size, self.config.EMBEDDING_DIM)
        ).astype(np.float32)

        # Set PAD embedding to zeros
        self.embedding_matrix[self.pad_idx] = np.zeros(self.config.EMBEDDING_DIM)

    def _set_vocab_from_list(self, vocab_list):
        """
        Helper to set internal dictionaries from a list of tokens.
        """
        self.idx_to_token = {i: token for i, token in enumerate(vocab_list)}
        self.token_to_idx = {token: i for i, token in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

    def _save_cache(self):
        """
        Saves the vocabulary list and embedding matrix to the cache directory.
        """
        # Ensure directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        vocab_path = self.config.VOCAB_PATH
        embedding_path = self.config.EMBEDDING_MATRIX_PATH

        # Convert idx_to_token to a list for saving as npy
        # This avoids using pickle for dictionaries
        vocab_list = [self.idx_to_token[i] for i in range(self.vocab_size)]

        print(f"Saving vocabulary to {vocab_path}...")
        np.save(vocab_path, np.array(vocab_list))

        print(f"Saving embedding matrix to {embedding_path}...")
        np.save(embedding_path, self.embedding_matrix)

    def text_to_indices(self, text):
        """
        Converts a string of text to a list of integer indices.

        Args:
            text (str): Input text.

        Returns:
            list[int]: List of token indices.
        """
        if not text:
            return []

        tokens = text.split()
        indices = [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        return indices

    def get_embedding_matrix(self):
        """
        Returns the embedding matrix.
        """
        if self.embedding_matrix is None:
            raise ValueError(
                "Embedding matrix not initialized. Call build_vocab() first."
            )
        return self.embedding_matrix

    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_token_idx(self):
        return self.pad_idx
