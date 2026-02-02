import os
import numpy as np
from collections import Counter
from library.config import Config


def ensure_dir(path):
    """
    Ensures that the directory for the given path exists.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

    def fit_on_texts(
        self, texts, min_freq=Config.MIN_FREQ, vocab_size=Config.VOCAB_SIZE
    ):
        """
        Builds vocabulary from a list of texts.
        """
        print("Fitting tokenizer on texts...")
        token_counts = Counter()

        for text in texts:
            if not isinstance(text, str):
                continue
            # Whitespace tokenization
            tokens = text.strip().split()
            token_counts.update(tokens)

        # Sort tokens by frequency (descending)
        most_common_tokens = token_counts.most_common()

        # Filter by min_freq and limit by vocab_size
        # We reserve 2 slots for PAD and UNK
        filtered_tokens = [
            token for token, count in most_common_tokens if count >= min_freq
        ]
        filtered_tokens = filtered_tokens[: vocab_size - 2]

        # Initialize dictionaries with special tokens
        self.word_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_word = {0: self.pad_token, 1: self.unk_token}

        # Populate dictionaries
        for i, token in enumerate(filtered_tokens):
            idx = i + 2
            self.word_index[token] = idx
            self.index_word[idx] = token

        self.vocab_size = len(self.word_index)
        print(f"Tokenizer fitted. Vocabulary Size: {self.vocab_size}")

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts to a list of sequences of integers.
        """
        sequences = []
        unk_idx = self.word_index.get(self.unk_token, 1)

        for text in texts:
            if not isinstance(text, str):
                sequences.append([])
                continue
            tokens = text.strip().split()
            seq = [self.word_index.get(token, unk_idx) for token in tokens]
            sequences.append(seq)

        return sequences

    def save(self, path):
        """
        Saves the vocabulary to a .npy file as an ordered list of strings.
        """
        ensure_dir(path)
        # Create a list where index i contains the word for ID i
        vocab_list = [self.index_word.get(i, "") for i in range(self.vocab_size)]
        np_vocab = np.array(vocab_list)
        np.save(path, np_vocab)
        print(f"Tokenizer vocabulary saved to {path}")

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        print(f"Loading tokenizer vocabulary from {path}...")
        # allow_pickle=True is often required for loading arrays of strings
        vocab_list = np.load(path, allow_pickle=True)

        self.word_index = {}
        self.index_word = {}

        for idx, word in enumerate(vocab_list):
            # Ensure word is a string
            word_str = str(word)
            self.word_index[word_str] = idx
            self.index_word[idx] = word_str

        self.vocab_size = len(self.word_index)
        print(f"Tokenizer loaded. Vocabulary Size: {self.vocab_size}")


def build_embedding_matrix(
    word_index, embedding_dim=Config.EMBEDDING_DIM, load_cached_data=True
):
    """
    Builds or loads the embedding matrix.

    Args:
        word_index: Dictionary mapping words to integers.
        embedding_dim: Dimension of the embeddings.
        load_cached_data: Boolean, whether to try loading from cache.

    Returns:
        embedding_matrix: Numpy array of shape (vocab_size, embedding_dim).
    """
    cache_path = Config.EMBEDDING_MATRIX_CACHE_FILE
    vocab_size = len(word_index)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embedding matrix from {cache_path}...")
        try:
            embedding_matrix = np.load(cache_path)
            if embedding_matrix.shape == (vocab_size, embedding_dim):
                return embedding_matrix
            else:
                print(
                    f"Cached embedding matrix shape {embedding_matrix.shape} mismatch with expected {(vocab_size, embedding_dim)}. Rebuilding..."
                )
        except Exception as e:
            print(f"Error loading cached embeddings: {e}. Rebuilding...")

    # 2. Compute from scratch
    print("Building embedding matrix...")

    # Initialize with random values (Normal distribution)
    # Using a fixed seed for reproducibility
    rng = np.random.RandomState(Config.SEED)
    embedding_matrix = rng.normal(scale=0.1, size=(vocab_size, embedding_dim))

    # Zero out the PAD token embedding
    if Config.PAD_TOKEN in word_index:
        pad_idx = word_index[Config.PAD_TOKEN]
        embedding_matrix[pad_idx] = np.zeros(embedding_dim)

    # Note: If pre-trained vectors (e.g., GloVe) were available at a specific path,
    # we would load them here and update the matrix rows matching the vocabulary.
    # Since no path is provided in Config, we proceed with random initialization.

    # 3. Save to cache
    ensure_dir(cache_path)
    np.save(cache_path, embedding_matrix)
    print(f"Embedding matrix saved to {cache_path}")

    return embedding_matrix
