import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config

# Define special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"


class Vocabulary:
    """
    Manages the mapping between text tokens and integer indices.
    """

    def __init__(self):
        self.stoi = {}  # String to Index
        self.itos = []  # Index to String
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN]

    def fit(self, texts, max_size=Config.VOCAB_SIZE):
        """
        Builds vocabulary from a list of text strings.

        Args:
            texts (iterable): List or Series of text strings.
            max_size (int): Maximum vocabulary size (excluding special tokens).
        """
        word_counts = Counter()
        for text in texts:
            tokens = clean_and_tokenize(text)
            word_counts.update(tokens)

        # Select most common words
        most_common = word_counts.most_common(max_size)

        # Initialize with special tokens
        self.itos = list(self.special_tokens)

        # Add common words
        for word, _ in most_common:
            self.itos.append(word)

        # Build stoi map
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def save(self, path):
        """
        Saves the vocabulary (itos list) to a numpy file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.itos))

    def load(self, path):
        """
        Loads the vocabulary from a numpy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        self.itos = np.load(path, allow_pickle=True).tolist()
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        """
        Converts a list of tokens to a list of indices.
        """
        unk_idx = self.stoi[UNK_TOKEN]
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def get_pad_index(self):
        return self.stoi[PAD_TOKEN]

    def get_mask_index(self):
        return self.stoi[MASK_TOKEN]


def clean_and_tokenize(text):
    """
    Cleans text and splits into tokens.

    Strategy:
    1. Lowercase.
    2. Extract sequences of alphanumeric characters.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    # Simple regex to keep words and numbers, ignoring punctuation
    tokens = re.findall(r"\w+", text)
    return tokens


def build_or_load_vocabulary(load_cached_data=True):
    """
    Builds the vocabulary from the training data or loads it from cache.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        Vocabulary: The initialized vocabulary object.
    """
    vocab_path = Config.VOCAB_SAVE_PATH

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    vocab = Vocabulary()

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}...")
        try:
            vocab.load(vocab_path)
            return vocab
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Building vocabulary from scratch...")

    # Load training data metadata to get text
    # We only need the text column for vocab building
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_PATH}")

    df_train = pd.read_csv(Config.TRAIN_PATH, usecols=[Config.TEXT_COL])
    texts = df_train[Config.TEXT_COL].fillna("").tolist()

    vocab.fit(texts, max_size=Config.VOCAB_SIZE)

    # Save to cache
    print(f"Saving vocabulary to {vocab_path}...")
    vocab.save(vocab_path)

    return vocab


def identify_identity_indices(vocab):
    """
    Identifies the vocabulary indices corresponding to the identity terms
    defined in Config.IDENTITY_KEYWORDS.

    Args:
        vocab (Vocabulary): The loaded vocabulary object.

    Returns:
        set: A set of integer indices corresponding to identity terms.
    """
    identity_indices = set()

    # Get the flattened set of keywords from Config
    target_terms = Config.get_identity_term_set()

    for term in target_terms:
        # Preprocess term to match tokenizer logic (lowercase)
        clean_term = term.lower()

        if clean_term in vocab.stoi:
            idx = vocab.stoi[clean_term]
            identity_indices.add(idx)

    return identity_indices
