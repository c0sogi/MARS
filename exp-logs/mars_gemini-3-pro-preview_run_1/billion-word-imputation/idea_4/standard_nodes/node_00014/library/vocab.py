import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("vocab")


class Vocabulary:
    """
    Handles tokenization, numericalization, and vocabulary management for the
    Interleaved Gap-Token Transformer.
    """

    def __init__(self):
        self.stoi = {}
        self.itos = []

        # Initialize with special tokens in specific order to match Config indices
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Adds special tokens defined in Config to the vocabulary."""
        special_tokens = [Config.PAD_TOKEN, Config.UNK_TOKEN, Config.GAP_TOKEN]
        for token in special_tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        # Verify indices match Config
        assert self.stoi[Config.PAD_TOKEN] == Config.PAD_IDX
        assert self.stoi[Config.UNK_TOKEN] == Config.UNK_IDX
        assert self.stoi[Config.GAP_TOKEN] == Config.GAP_IDX

    def __len__(self):
        return len(self.itos)

    def add_token(self, token):
        """Adds a token to the vocabulary if it doesn't exist."""
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def build(self, sentences, max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ):
        """
        Builds the vocabulary from a list of sentences.

        Args:
            sentences (list of str): List of training sentences.
            max_size (int): Maximum vocabulary size (excluding special tokens).
            min_freq (int): Minimum frequency for a word to be included.
        """
        logger.info("Tokenizing and counting word frequencies...")
        counter = Counter()
        for sentence in sentences:
            # Simple whitespace tokenization based on dataset description
            tokens = str(sentence).split()
            counter.update(tokens)

        logger.info(f"Total unique tokens found: {len(counter)}")

        # Filter by frequency and size
        # We reserve space for existing special tokens
        current_vocab_size = len(self.itos)
        limit = max_size - current_vocab_size

        most_common = counter.most_common()
        added_count = 0

        for word, freq in most_common:
            if added_count >= limit:
                break
            if freq < min_freq:
                break

            if word not in self.stoi:
                self.add_token(word)
                added_count += 1

        logger.info(f"Vocabulary built. Size: {len(self.itos)}")

    def save(self, path):
        """Saves the vocabulary (itos list) to a numpy file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.itos))
        logger.info(f"Vocabulary saved to {path}")

    def load(self, path):
        """Loads the vocabulary from a numpy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        self.itos = np.load(path).tolist()
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        logger.info(f"Vocabulary loaded from {path}. Size: {len(self.itos)}")

    def encode(self, text):
        """
        Converts a sentence string into a list of token indices.

        Args:
            text (str): Input sentence.

        Returns:
            list[int]: List of token indices.
        """
        tokens = str(text).split()
        return [self.stoi.get(token, Config.UNK_IDX) for token in tokens]

    def encode_tokens(self, tokens):
        """
        Converts a list of tokens into a list of token indices.

        Args:
            tokens (list[str]): List of tokens.

        Returns:
            list[int]: List of token indices.
        """
        return [self.stoi.get(token, Config.UNK_IDX) for token in tokens]

    def decode(self, indices):
        """
        Converts a list of indices back to a sentence string.

        Args:
            indices (list[int]): List of token indices.

        Returns:
            str: Reconstructed sentence.
        """
        tokens = []
        for idx in indices:
            if 0 <= idx < len(self.itos):
                tokens.append(self.itos[idx])
            else:
                tokens.append(Config.UNK_TOKEN)
        return " ".join(tokens)


def get_vocab(
    load_cached_data=True,
    vocab_path=Config.VOCAB_PATH,
    train_metadata_path=Config.TRAIN_METADATA_PATH,
):
    """
    Factory function to get the vocabulary. Implements caching logic.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        vocab_path (str): Path to the saved vocabulary file.
        train_metadata_path (str): Path to training metadata for rebuilding.

    Returns:
        Vocabulary: The initialized vocabulary object.
    """
    vocab = Vocabulary()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(vocab_path):
        logger.info(f"Cache hit: Loading vocabulary from {vocab_path}")
        try:
            vocab.load(vocab_path)
            return vocab
        except Exception as e:
            logger.warning(f"Failed to load cached vocabulary: {e}. Rebuilding...")

    # Rebuild if cache miss or load failed
    logger.info("Cache miss or force rebuild: Building vocabulary from scratch...")

    if not os.path.exists(train_metadata_path):
        raise FileNotFoundError(f"Training metadata not found at {train_metadata_path}")

    # Load training data
    logger.info(f"Loading training sentences from {train_metadata_path}")
    df = pd.read_csv(train_metadata_path)

    # Handle potential missing values
    sentences = df["sentence"].dropna().astype(str).tolist()

    # Build vocabulary
    vocab.build(sentences, max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ)

    # Save to cache
    vocab.save(vocab_path)

    return vocab
