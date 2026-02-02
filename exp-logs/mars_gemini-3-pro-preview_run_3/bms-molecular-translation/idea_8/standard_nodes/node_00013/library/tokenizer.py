import json
import os
import pandas as pd
from library.config import Config


class Tokenizer:
    """
    Tokenizer class for converting InChI text strings into sequences of integer indices
    and vice versa. It manages the vocabulary, including special tokens, and supports
    caching to ensure consistency and speed across runs.
    """

    def __init__(self, config: Config, load_cached_data: bool = True):
        """
        Initialize the Tokenizer.

        Args:
            config (Config): Configuration object containing paths and special tokens.
            load_cached_data (bool): Whether to try loading the vocabulary from cache.
        """
        self.config = config
        self.stoi = {}
        self.itos = {}

        # Special tokens
        self.sos_token = config.sos_token
        self.eos_token = config.eos_token
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token

        self.load_or_build_vocab(load_cached_data)

    def load_or_build_vocab(self, load_cached_data: bool):
        """
        Loads the vocabulary from a cached JSON file or builds it from the training metadata.

        Args:
            load_cached_data (bool): If True, attempts to load from config.vocab_path.
        """
        vocab_path = self.config.vocab_path

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            try:
                with open(vocab_path, "r") as f:
                    self.stoi = json.load(f)
                # Reconstruct itos (values in stoi are ints, so they become keys here)
                self.itos = {v: k for k, v in self.stoi.items()}
                return
            except Exception as e:
                print(f"Failed to load vocabulary: {e}. Rebuilding...")

        # 2. Build from scratch
        print("Building vocabulary from training data...")
        if not os.path.exists(self.config.train_metadata_path):
            raise FileNotFoundError(
                f"Training metadata not found at {self.config.train_metadata_path}"
            )

        df = pd.read_csv(self.config.train_metadata_path)

        # Extract unique characters from the InChI column
        # Using unique() on the series first is much faster than iterating over all rows
        unique_inchis = df["InChI"].unique()
        unique_chars = set()

        # Process in chunks to avoid creating massive temporary strings
        chunk_size = 100000
        for i in range(0, len(unique_inchis), chunk_size):
            chunk = unique_inchis[i : i + chunk_size]
            text_blob = "".join(chunk)
            unique_chars.update(text_blob)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Define indices
        # Special tokens are assigned the first few indices
        special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]

        self.stoi = {token: i for i, token in enumerate(special_tokens)}

        # Add dataset characters
        start_idx = len(special_tokens)
        for i, char in enumerate(sorted_chars):
            self.stoi[char] = start_idx + i

        self.itos = {i: c for c, i in self.stoi.items()}

        # 3. Save to cache
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(self.stoi, f, indent=4)
        print(f"Vocabulary saved to {vocab_path}")

    def text_to_sequence(self, text: str) -> list:
        """
        Converts a raw InChI string to a sequence of integer indices.
        Wraps the sequence with <sos> and <eos> tokens.

        Args:
            text (str): The InChI string.

        Returns:
            list: List of integer indices.
        """
        sequence = [self.stoi[self.sos_token]]
        for char in text:
            sequence.append(self.stoi.get(char, self.stoi[self.unk_token]))
        sequence.append(self.stoi[self.eos_token])
        return sequence

    def sequence_to_text(self, sequence: list) -> str:
        """
        Converts a sequence of integer indices back to a string.
        Stops decoding when <eos> is encountered. Ignores <sos> and <pad>.

        Args:
            sequence (list): List of integer indices (or tensor/numpy array).

        Returns:
            str: The decoded InChI string.
        """
        result = []
        for idx in sequence:
            # Handle PyTorch tensors or NumPy scalars if passed
            if hasattr(idx, "item"):
                idx = idx.item()

            # Retrieve token, default to unk if out of bounds (though unlikely with valid vocab)
            token = self.itos.get(idx, self.unk_token)

            if token == self.eos_token:
                break
            if token == self.sos_token or token == self.pad_token:
                continue

            result.append(token)

        return "".join(result)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.stoi)
