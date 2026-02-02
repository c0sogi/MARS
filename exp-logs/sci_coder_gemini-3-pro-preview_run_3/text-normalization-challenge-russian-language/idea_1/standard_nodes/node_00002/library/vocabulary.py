import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class CharVocab:
    """
    Vocabulary class to handle character-to-index and index-to-character mapping.
    """

    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        # Special token indices
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # Mappings
        self.char2idx = {}
        self.idx2char = {}

    def build_vocab(self, data_path=Config.TRAIN_DATA_PATH, load_cached_data=True):
        """
        Builds the vocabulary from the provided dataset or loads it from cache.

        Args:
            data_path (str): Path to the training CSV file.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        vocab_path = Config.VOCAB_PATH
        cache_dir = os.path.dirname(vocab_path)

        # Ensure working directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(vocab_path):
            try:
                print(f"Loading vocabulary from {vocab_path}...")
                vocab_array = np.load(vocab_path)
                self._create_maps_from_array(vocab_array)
                print(f"Vocabulary loaded. Size: {len(self)}")
                return
            except Exception as e:
                print(
                    f"Failed to load cached vocabulary: {e}. Rebuilding from scratch..."
                )

        # 2. Build from scratch
        print(f"Building vocabulary from {data_path}...")

        # Load data (only text columns needed)
        try:
            df = pd.read_csv(data_path, usecols=["before", "after"], dtype=str)
        except ValueError:
            # Fallback if specific columns aren't found, load all
            df = pd.read_csv(data_path, dtype=str)

        unique_chars = set()

        # Extract characters from 'before' column
        if "before" in df.columns:
            # Drop NaNs and concatenate all text
            text_before = "".join(df["before"].dropna().tolist())
            unique_chars.update(text_before)

        # Extract characters from 'after' column
        if "after" in df.columns:
            text_after = "".join(df["after"].dropna().tolist())
            unique_chars.update(text_after)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Prepend special tokens
        full_vocab_list = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ] + sorted_chars
        vocab_array = np.array(full_vocab_list)

        # Save to cache (using numpy format, no pickle module usage)
        np.save(vocab_path, vocab_array)

        # Create mappings
        self._create_maps_from_array(vocab_array)
        print(f"Vocabulary built and saved to {vocab_path}. Size: {len(self)}")

    def _create_maps_from_array(self, vocab_array):
        """
        Helper to populate dictionaries from the vocabulary array.
        """
        self.idx2char = {i: char for i, char in enumerate(vocab_array)}
        self.char2idx = {char: i for i, char in enumerate(vocab_array)}

    def encode(self, text, add_special_tokens=True):
        """
        Converts a string to a list of indices.

        Args:
            text (str): Input text.
            add_special_tokens (bool): Whether to add SOS and EOS tokens.

        Returns:
            list[int]: List of token indices.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        indices = []
        if add_special_tokens:
            indices.append(self.sos_idx)

        for char in text:
            indices.append(self.char2idx.get(char, self.unk_idx))

        if add_special_tokens:
            indices.append(self.eos_idx)

        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a list or tensor of indices back to a string.

        Args:
            indices (list or torch.Tensor): Input indices.
            remove_special_tokens (bool): Whether to remove special tokens and stop at EOS.

        Returns:
            str: Decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            # If removing specials, handle EOS and skip others
            if remove_special_tokens:
                if idx == self.eos_idx:
                    break
                if idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                    continue

            # Map index to char
            chars.append(self.idx2char.get(idx, self.unk_token))

        return "".join(chars)

    def __len__(self):
        return len(self.char2idx)
