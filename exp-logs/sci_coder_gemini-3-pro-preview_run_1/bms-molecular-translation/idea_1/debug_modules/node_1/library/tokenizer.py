import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, encoding, decoding, and caching.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        # Special tokens are defined in Config, but we assign specific indices here
        # to ensure consistency.
        self.pad_token = Config.PAD_TOKEN
        self.sos_token = Config.SOS_TOKEN
        self.eos_token = Config.EOS_TOKEN
        self.unk_token = Config.UNK_TOKEN

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # Initialize vocab with special tokens
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self._update_mappings()

    def _update_mappings(self):
        """Helper to update dictionary mappings from the list vocabulary."""
        self.idx2char = {i: char for i, char in enumerate(self.vocab)}
        self.char2idx = {char: i for i, char in enumerate(self.vocab)}

    def build_vocab(self, load_cached_data=True):
        """
        Builds the vocabulary from the training metadata.
        Implements caching using .npy files.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
        """
        cache_path = Config.TOKENIZER_PATH

        # Ensure the directory exists (redundant if Config handles it, but safe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading tokenizer vocabulary from {cache_path}...")
                self.vocab = np.load(cache_path).tolist()
                self._update_mappings()
                loaded = True
            except Exception as e:
                print(f"Failed to load cached tokenizer: {e}. Rebuilding...")
                loaded = False

        if not loaded:
            print("Building tokenizer vocabulary from source data...")
            # Load training metadata
            if not os.path.exists(Config.TRAIN_METADATA_PATH):
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
                )

            df = pd.read_csv(Config.TRAIN_METADATA_PATH)

            # Extract all unique characters from InChI strings
            # We assume the column name is 'InChI' based on metadata generation
            if "InChI" not in df.columns:
                raise ValueError("Column 'InChI' not found in training metadata.")

            # Use a set for unique characters
            chars = set()
            # Iterate through the series to avoid massive memory overhead of joining all strings
            for text in df["InChI"].astype(str):
                chars.update(text)

            # Sort characters for deterministic behavior
            sorted_chars = sorted(list(chars))

            # Re-initialize vocab with special tokens followed by sorted characters
            self.vocab = [
                self.pad_token,
                self.sos_token,
                self.eos_token,
                self.unk_token,
            ] + sorted_chars
            self._update_mappings()

            # Save to cache
            print(f"Saving tokenizer vocabulary to {cache_path}...")
            np.save(cache_path, np.array(self.vocab))

        print(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        """
        Converts an InChI string to a tensor of indices with padding.
        Adds <sos> at the start and <eos> at the end.

        Args:
            text (str): The InChI string.

        Returns:
            torch.LongTensor: Tensor of shape (MAX_LEN,) containing indices.
        """
        # Start with SOS
        indices = [self.sos_idx]

        # Convert characters
        for char in str(text):
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.unk_idx)

        # Add EOS
        indices.append(self.eos_idx)

        # Padding
        max_len = Config.MAX_LEN
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        else:
            # Truncate if too long (though MAX_LEN should cover it)
            # We ensure EOS is preserved at the end if truncated,
            # but ideally MAX_LEN is sufficient.
            indices = indices[: max_len - 1] + [self.eos_idx]

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """
        Converts a sequence of indices back to a string.
        Stops at <eos> and ignores <pad>, <sos>, <unk>.

        Args:
            indices (list or torch.Tensor): Sequence of integer indices.

        Returns:
            str: Decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx:
                continue
            if idx == self.sos_idx:
                continue

            # We keep UNK as a placeholder or ignore it.
            # Usually ignoring it or printing '?' is fine.
            # Here we simply append the mapped char.
            if idx in self.idx2char:
                tokens.append(self.idx2char[idx])
            else:
                # Should not happen if vocab is consistent
                pass

        return "".join(tokens)

    def __len__(self):
        return len(self.vocab)
