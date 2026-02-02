import os
import pandas as pd
import numpy as np
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles special tokens <PAD>, <SOS>, <EOS> and vocabulary management.
    """

    def __init__(self, load_cached_data=True, debug=False):
        """
        Initialize the Tokenizer.

        Args:
            load_cached_data (bool): Whether to try loading the vocabulary from cache.
            debug (bool): If True, use a subset of data for vocabulary building (if not cached).
        """
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

        self.stoi = {}
        self.itos = {}
        self.tokens = []

        self._load_or_build_vocab(load_cached_data, debug)

    def _load_or_build_vocab(self, load_cached_data, debug):
        """
        Implements the caching logic for vocabulary construction.
        """
        cache_path = Config.TOKENIZER_PATH
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        vocab_loaded = False

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                try:
                    print(f"Loading tokenizer vocabulary from {cache_path}...")
                    self.tokens = np.load(cache_path)
                    vocab_loaded = True
                except Exception as e:
                    print(f"Failed to load cached tokenizer: {e}")
            else:
                print(f"Cached tokenizer not found at {cache_path}")

        # 2. Build if not loaded
        if not vocab_loaded:
            print("Building vocabulary from training metadata...")

            if not os.path.exists(Config.TRAIN_METADATA):
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA}"
                )

            # Read CSV
            # Use debug flag to control dataset size for faster vocab building during dev
            if debug:
                print("Debug mode: Loading subset of data for vocab building.")
                df = pd.read_csv(Config.TRAIN_METADATA, nrows=10000)
            else:
                df = pd.read_csv(Config.TRAIN_METADATA)

            # Extract unique characters from the InChI column
            all_inchi = df["InChI"].astype(str).tolist()
            unique_chars = set()
            for inchi in all_inchi:
                unique_chars.update(inchi)

            # Sort for determinism
            sorted_chars = sorted(list(unique_chars))

            # Combine with special tokens
            self.tokens = np.array(self.special_tokens + sorted_chars)

            # Save to cache
            print(f"Saving vocabulary to {cache_path}...")
            np.save(cache_path, self.tokens)

        # Build mappings
        self.itos = {i: token for i, token in enumerate(self.tokens)}
        self.stoi = {token: i for i, token in enumerate(self.tokens)}

        print(f"Vocabulary size: {len(self.tokens)}")
        print(f"Special tokens: {self.special_tokens}")

    def text_to_sequence(self, text, max_len=None, padding=False):
        """
        Converts an InChI string to a sequence of integers.
        Adds <SOS> at the start and <EOS> at the end.

        Args:
            text (str): Input InChI string.
            max_len (int, optional): Maximum length for padding/truncation.
            padding (bool): Whether to pad the sequence to max_len.

        Returns:
            list[int]: Sequence of token indices.
        """
        sequence = [self.stoi[self.sos_token]]

        for char in text:
            if char in self.stoi:
                sequence.append(self.stoi[char])
            # We silently ignore unknown characters as the vocab covers the training set

        sequence.append(self.stoi[self.eos_token])

        if padding and max_len is not None:
            # Truncate if too long (reserving space for EOS)
            if len(sequence) > max_len:
                sequence = sequence[: max_len - 1] + [self.stoi[self.eos_token]]

            # Pad if too short
            if len(sequence) < max_len:
                sequence += [self.stoi[self.pad_token]] * (max_len - len(sequence))

        return sequence

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of integers back to an InChI string.
        Stops at <EOS>. Ignores <SOS> and <PAD>.

        Args:
            sequence (list[int] or torch.Tensor): Input sequence.

        Returns:
            str: Decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        result = []
        for idx in sequence:
            idx = int(idx)
            # Handle potential out-of-bounds indices safely
            token = self.itos.get(idx, "")

            if token == self.sos_token:
                continue
            if token == self.pad_token:
                continue
            if token == self.eos_token:
                break

            result.append(token)

        return "".join(result)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.tokens)

    @property
    def pad_token_id(self):
        return self.stoi[self.pad_token]

    @property
    def sos_token_id(self):
        return self.stoi[self.sos_token]

    @property
    def eos_token_id(self):
        return self.stoi[self.eos_token]
