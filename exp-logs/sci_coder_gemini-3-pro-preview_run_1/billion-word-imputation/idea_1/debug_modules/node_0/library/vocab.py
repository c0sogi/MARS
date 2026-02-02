import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config
from library.utils import logger


class Vocabulary:
    """
    Manages the vocabulary for the Sentence Infilling Task.
    Handles tokenization, numericalization, and storage of the vocabulary.
    """

    def __init__(self):
        self.itos = []  # Index to String
        self.stoi = {}  # String to Index

        # Define special tokens
        self.TOKEN_PAD = Config.TOKEN_PAD
        self.TOKEN_UNK = Config.TOKEN_UNK
        self.TOKEN_NO_INSERT = Config.TOKEN_NO_INSERT
        self.TOKEN_START = "[START]"
        self.TOKEN_END = "[END]"

        # Order ensures fixed indices for these tokens
        self.specials = [
            self.TOKEN_PAD,
            self.TOKEN_UNK,
            self.TOKEN_NO_INSERT,
            self.TOKEN_START,
            self.TOKEN_END,
        ]

    def build_from_corpus(self, load_cached_data=True):
        """
        Builds the vocabulary from the training corpus or loads it from cache.

        Args:
            load_cached_data (bool): If True, attempts to load from Config.VOCAB_FILE.
                                     If False or file missing, rebuilds from scratch.
        """
        vocab_path = Config.VOCAB_FILE

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(vocab_path):
            logger.info(f"Loading vocabulary from cache: {vocab_path}")
            try:
                self.load(vocab_path)
                logger.info(f"Vocabulary loaded. Size: {len(self)}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached vocabulary: {e}. Rebuilding...")

        # 2. Rebuild from scratch
        logger.info("Building vocabulary from corpus...")

        # Ensure working directory exists
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

        # Initialize counter
        token_counts = Counter()

        # Read training data
        train_path = Config.TRAIN_METADATA_PATH
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training metadata not found at {train_path}")

        # Determine chunk size for memory efficiency
        chunk_size = 100_000

        # Handle debug sampling
        nrows = Config.DEBUG_SAMPLE_SIZE

        logger.info(f"Reading data from {train_path} (nrows={nrows})...")

        try:
            # Use pandas to read in chunks
            reader = pd.read_csv(train_path, chunksize=chunk_size, nrows=nrows)

            for i, chunk in enumerate(reader):
                # Filter out null sentences
                sentences = chunk["sentence"].dropna().astype(str).tolist()

                # Tokenize and update counter
                # Simple whitespace tokenization as per analysis
                for sentence in sentences:
                    tokens = sentence.split()
                    token_counts.update(tokens)

                if (i + 1) * chunk_size % 1_000_000 == 0:
                    logger.info(f"Processed {(i + 1) * chunk_size} rows...")

        except Exception as e:
            logger.error(f"Error reading corpus: {e}")
            raise

        logger.info(f"Total unique tokens found: {len(token_counts)}")

        # Select top frequent words
        # Available slots for words = VOCAB_SIZE - len(specials)
        max_words = Config.VOCAB_SIZE - len(self.specials)
        most_common = token_counts.most_common(max_words)

        # Construct vocabulary
        self.itos = list(self.specials)
        for word, freq in most_common:
            self.itos.append(word)

        # Rebuild stoi
        self.stoi = {word: i for i, word in enumerate(self.itos)}

        logger.info(f"Vocabulary built. Final size: {len(self)}")

        # 3. Save to cache
        self.save(vocab_path)

    def save(self, path):
        """
        Saves the vocabulary (itos list) to a .npy file.
        """
        logger.info(f"Saving vocabulary to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.itos))

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        itos_array = np.load(path)
        self.itos = itos_array.tolist()
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        """Returns the index of the token, or UNK index if not found."""
        return self.stoi.get(token, self.stoi.get(self.TOKEN_UNK))

    def get_token(self, index):
        """Returns the token at the given index."""
        if 0 <= index < len(self.itos):
            return self.itos[index]
        return self.TOKEN_UNK

    def encode(self, sentence, add_special_tokens=False):
        """
        Converts a sentence string into a list of token indices.

        Args:
            sentence (str): The input sentence.
            add_special_tokens (bool): If True, wraps with [START] and [END].

        Returns:
            list[int]: List of token indices.
        """
        tokens = sentence.split()
        indices = [self[token] for token in tokens]

        if add_special_tokens:
            start_idx = self.stoi[self.TOKEN_START]
            end_idx = self.stoi[self.TOKEN_END]
            indices = [start_idx] + indices + [end_idx]

        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a list of indices back into a sentence string.

        Args:
            indices (list[int] or np.ndarray): List of token indices.
            remove_special_tokens (bool): If True, removes [START], [END], [PAD], [NO_INSERT].

        Returns:
            str: The reconstructed sentence.
        """
        tokens = []
        for idx in indices:
            token = self.get_token(idx)

            if remove_special_tokens:
                if token in {
                    self.TOKEN_PAD,
                    self.TOKEN_START,
                    self.TOKEN_END,
                    self.TOKEN_NO_INSERT,
                }:
                    continue

            tokens.append(token)

        return " ".join(tokens)

    @property
    def pad_token_id(self):
        return self.stoi[self.TOKEN_PAD]

    @property
    def unk_token_id(self):
        return self.stoi[self.TOKEN_UNK]

    @property
    def no_insert_token_id(self):
        return self.stoi[self.TOKEN_NO_INSERT]
