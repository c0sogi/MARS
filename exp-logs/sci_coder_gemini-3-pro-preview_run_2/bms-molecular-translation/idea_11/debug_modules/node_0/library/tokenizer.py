import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class InChITokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, caching, and special tokens for autoregressive modeling.
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initialize the tokenizer.

        Args:
            load_cached_data (bool): If True, attempts to load the vocabulary from cache.
                                     If False or cache not found, builds from scratch.
        """
        self.stoi = {}
        self.itos = {}

        # Define special tokens
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"

        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2

        # Initial mapping with special tokens
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

        # Load or build vocabulary
        self._load_or_build_vocab(load_cached_data)

    def _load_or_build_vocab(self, load_cached_data: bool):
        """
        Loads vocabulary from cache or builds it from the training metadata.
        Strictly follows the caching logic requirement.
        """
        vocab_path = Config.VOCAB_PATH
        cache_exists = os.path.exists(vocab_path)

        if load_cached_data and cache_exists:
            print(f"Loading vocabulary from {vocab_path}...")
            try:
                vocab_chars = np.load(vocab_path)
                self._create_mappings(vocab_chars)
                return
            except Exception as e:
                print(f"Failed to load cached vocabulary: {e}. Rebuilding...")

        print("Building vocabulary from training metadata...")
        self._build_vocab_from_metadata()

    def _build_vocab_from_metadata(self):
        """
        Reads the training metadata, extracts unique characters, and creates mappings.
        Saves the resulting vocabulary to cache.
        """
        train_csv_path = Config.TRAIN_METADATA
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training metadata not found at {train_csv_path}")

        df = pd.read_csv(train_csv_path)

        # Extract all unique characters from the InChI column
        # Using a set for uniqueness
        unique_chars = set()
        # We assume the InChI column exists and contains strings
        # Iterate efficiently
        for text in df["InChI"].astype(str):
            unique_chars.update(text)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Combine special tokens and actual characters
        full_vocab = np.array(self.special_tokens + sorted_chars)

        # Create mappings
        self._create_mappings(full_vocab)

        # Save to cache
        os.makedirs(os.path.dirname(Config.VOCAB_PATH), exist_ok=True)
        np.save(Config.VOCAB_PATH, full_vocab)
        print(f"Vocabulary saved to {Config.VOCAB_PATH}. Size: {len(full_vocab)}")

    def _create_mappings(self, vocab_chars):
        """
        Populate stoi and itos dictionaries from a list of characters.
        """
        self.itos = {i: char for i, char in enumerate(vocab_chars)}
        self.stoi = {char: i for i, char in enumerate(vocab_chars)}

    def text_to_sequence(self, text: str):
        """
        Converts an InChI string to a sequence of integers.
        Adds SOS at the beginning and EOS at the end.

        Args:
            text (str): The InChI string.

        Returns:
            List[int]: The sequence of indices.
        """
        sequence = [self.SOS_IDX]
        for char in text:
            if char in self.stoi:
                sequence.append(self.stoi[char])
            else:
                # In a real scenario, we might handle unknown chars,
                # but for this closed dataset, we assume coverage.
                # Could optionally print a warning or skip.
                pass
        sequence.append(self.EOS_IDX)
        return sequence

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of integers back to an InChI string.
        Stops at EOS. Ignores PAD and SOS.

        Args:
            sequence (List[int] or torch.Tensor): The sequence of indices.

        Returns:
            str: The decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()

        result = []
        for idx in sequence:
            if idx == self.EOS_IDX:
                break
            if idx == self.PAD_IDX or idx == self.SOS_IDX:
                continue

            if idx in self.itos:
                result.append(self.itos[idx])

        return "".join(result)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.stoi)
