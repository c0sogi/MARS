import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Handles conversion between InChI text strings and integer indices for CTC loss.
    Manages vocabulary creation, caching, encoding, and greedy decoding.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the tokenizer.

        Args:
            load_cached_data (bool): If True, attempts to load the vocabulary from cache.
                                     If False or load fails, rebuilds from training metadata.
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab = []
        self.blank_token_idx = 0

        # Ensure the working directory exists as per requirements
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self._build_vocab(load_cached_data)

    def _build_vocab(self, load_cached_data):
        """
        Builds the vocabulary from training metadata or loads it from a cached file.
        """
        vocab_path = Config.VOCAB_PATH
        loaded = False

        # Attempt to load from cache
        if load_cached_data:
            if os.path.exists(vocab_path):
                try:
                    self.vocab = np.load(vocab_path).tolist()
                    print(f"Loaded vocabulary from {vocab_path}")
                    loaded = True
                except Exception as e:
                    print(f"Failed to load vocabulary from {vocab_path}: {e}")
            else:
                print(f"Cached vocabulary not found at {vocab_path}")

        # Build from scratch if not loaded
        if not loaded:
            print(f"Building vocabulary from {Config.TRAIN_METADATA_PATH}")
            if not os.path.exists(Config.TRAIN_METADATA_PATH):
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
                )

            df = pd.read_csv(Config.TRAIN_METADATA_PATH)

            # Extract unique characters from the InChI column
            unique_chars = set()
            # Ensure we are iterating over strings
            for text in df["InChI"].astype(str):
                unique_chars.update(text)

            # Sort for determinism
            self.vocab = sorted(list(unique_chars))

            # Save to cache
            try:
                np.save(vocab_path, np.array(self.vocab))
                print(f"Vocabulary saved to {vocab_path}")
            except Exception as e:
                print(f"Warning: Could not save vocabulary cache: {e}")

        # Construct mappings
        # Index 0 is reserved for the CTC blank token
        # Real characters start from index 1
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.vocab)}
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.vocab)}

        # Explicitly map blank token for reverse lookup (maps to empty string)
        self.idx_to_char[self.blank_token_idx] = ""

    def __len__(self):
        """
        Returns the size of the vocabulary including the blank token.
        """
        return len(self.vocab) + 1

    def encode(self, text):
        """
        Converts a string to a tensor of indices.

        Args:
            text (str): The InChI string.

        Returns:
            torch.LongTensor: Tensor of indices.
        """
        # Filter out characters not in vocabulary to prevent errors
        indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return torch.LongTensor(indices)

    def decode_greedy(self, logits):
        """
        Decodes logits using CTC greedy decoding logic.

        Args:
            logits (torch.Tensor): Tensor of shape (Batch, Time, Classes).

        Returns:
            List[str]: List of decoded InChI strings.
        """
        # Ensure logits are on CPU for numpy-like iteration
        if logits.is_cuda:
            logits = logits.detach().cpu()

        # Get argmax indices along the class dimension
        # Shape: (Batch, Time)
        predictions = torch.argmax(logits, dim=-1)

        decoded_strings = []

        for seq in predictions:
            decoded_chars = []
            prev_idx = -1

            for idx in seq:
                idx = idx.item()

                # CTC Greedy Decoding Logic:
                # 1. Collapse consecutive duplicates: If current index is same as previous, skip.
                if idx == prev_idx:
                    continue

                # 2. Remove blanks: If current index is blank (0), skip (but update prev_idx).
                if idx != self.blank_token_idx:
                    decoded_chars.append(self.idx_to_char.get(idx, ""))

                # Update previous index
                prev_idx = idx

            decoded_strings.append("".join(decoded_chars))

        return decoded_strings
