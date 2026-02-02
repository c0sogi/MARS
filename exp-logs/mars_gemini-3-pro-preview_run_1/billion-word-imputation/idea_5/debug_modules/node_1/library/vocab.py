import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Union

from library.config import Config
from library.utils import get_logger


class Vocabulary:
    """
    Manages the vocabulary for the Global-Localization Interleaved Transformer.
    Handles mapping between string tokens and integer indices, including special tokens.
    Implements strict caching using .npy files.
    """

    def __init__(self):
        self.logger = get_logger("Vocabulary")

        # Special tokens from Config
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN
        self.gap_token = Config.GAP_TOKEN

        # Mappings
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

        # Indices for special tokens (initialized after build/load)
        self.pad_index = -1
        self.unk_index = -1
        self.gap_index = -1

    def build(self, load_cached_data: bool = True) -> None:
        """
        Builds the vocabulary from the training corpus or loads it from cache.

        Args:
            load_cached_data (bool): If True, attempts to load from Config.VOCAB_PATH.
                                     If False or load fails, builds from scratch.
        """
        vocab_path = Config.VOCAB_PATH

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(vocab_path):
            self.logger.info(f"Loading vocabulary from cache: {vocab_path}")
            try:
                self._load_from_file(vocab_path)
                self.logger.info(
                    f"Vocabulary loaded successfully. Size: {len(self.itos)}"
                )
                return
            except Exception as e:
                self.logger.warning(
                    f"Failed to load vocabulary from cache: {e}. Rebuilding..."
                )

        # 2. Build from scratch
        self.logger.info("Building vocabulary from scratch...")

        # Ensure working directory exists
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

        # Load training data
        train_csv_path = Config.TRAIN_METADATA_PATH
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training metadata not found at {train_csv_path}")

        df = pd.read_csv(train_csv_path)

        # Handle Debugging
        if Config.DEBUG:
            self.logger.info(f"Debug mode enabled. Sampling {Config.DEBUG_SIZE} rows.")
            df = df.head(Config.DEBUG_SIZE)

        sentences = df["sentence"].astype(str).tolist()

        # Count frequencies
        counter = Counter()
        for sentence in sentences:
            tokens = sentence.split()
            counter.update(tokens)

        self.logger.info(f"Total unique tokens found: {len(counter)}")

        # Select top frequent words
        # Reserve slots for special tokens: PAD, UNK, GAP
        num_special_tokens = 3
        max_vocab_words = Config.VOCAB_SIZE - num_special_tokens

        most_common = counter.most_common(max_vocab_words)

        # Construct vocabulary list (itos)
        # Order: [PAD, UNK, GAP, ...frequent words...]
        self.itos = [self.pad_token, self.unk_token, self.gap_token]
        self.itos.extend([token for token, count in most_common])

        # Construct dictionary (stoi)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        # Update indices
        self._update_indices()

        self.logger.info(f"Vocabulary built. Final size: {len(self.itos)}")

        # 3. Save to cache
        self._save_to_file(vocab_path)

    def _save_to_file(self, path: str) -> None:
        """Saves the vocabulary list to a .npy file."""
        try:
            np.save(path, np.array(self.itos))
            self.logger.info(f"Vocabulary saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving vocabulary: {e}")
            raise

    def _load_from_file(self, path: str) -> None:
        """Loads the vocabulary list from a .npy file and reconstructs mappings."""
        self.itos = np.load(path).tolist()
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self._update_indices()

    def _update_indices(self) -> None:
        """Updates the stored indices for special tokens."""
        self.pad_index = self.stoi.get(self.pad_token, -1)
        self.unk_index = self.stoi.get(self.unk_token, -1)
        self.gap_index = self.stoi.get(self.gap_token, -1)

        # Sanity check
        if -1 in [self.pad_index, self.unk_index, self.gap_index]:
            self.logger.warning(
                "One or more special tokens are missing from the vocabulary!"
            )

    def encode(self, token: str) -> int:
        """
        Converts a single string token to its integer ID.
        Returns UNK index if token is not found.
        """
        return self.stoi.get(token, self.unk_index)

    def decode(self, idx: int) -> str:
        """
        Converts a single integer ID to its string token.
        Returns UNK token if ID is out of bounds.
        """
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return self.unk_token

    def encode_sequence(self, tokens: List[str]) -> List[int]:
        """Converts a list of tokens to a list of IDs."""
        return [self.encode(t) for t in tokens]

    def decode_sequence(self, indices: List[int]) -> List[str]:
        """Converts a list of IDs to a list of tokens."""
        return [self.decode(idx) for idx in indices]

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.itos)
