import os
import json
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Handles conversion between InChI text strings and integer sequences.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the Tokenizer.

        Args:
            load_cached_data (bool): Whether to try loading the vocabulary from cache.
        """
        self.stoi = {}
        self.itos = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]

        # Build or load vocabulary
        self.build_vocab(load_cached_data=load_cached_data)

        self.pad_token_id = self.stoi[self.pad_token]
        self.sos_token_id = self.stoi[self.sos_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.unk_token_id = self.stoi[self.unk_token]

    def build_vocab(self, load_cached_data):
        """
        Builds the vocabulary from the training metadata or loads it from cache.

        Args:
            load_cached_data (bool): If True, attempts to load from Config.VOCAB_PATH.
        """
        vocab_path = Config.VOCAB_PATH

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            try:
                with open(vocab_path, "r") as f:
                    vocab_data = json.load(f)
                self.stoi = vocab_data["stoi"]
                self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
                return
            except Exception as e:
                print(f"Failed to load cached vocabulary: {e}. Rebuilding...")

        # 2. Build from scratch
        print("Building vocabulary from training metadata...")
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
            )

        df = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # Extract all unique characters from InChI strings
        # We use a set to collect unique characters
        unique_chars = set()
        # Iterate in chunks to be memory efficient if needed, though InChI dataset fits in memory
        for text in df["InChI"].dropna():
            unique_chars.update(text)

        sorted_chars = sorted(list(unique_chars))

        # Create mappings
        # Add special tokens first
        self.stoi = {token: i for i, token in enumerate(self.special_tokens)}

        # Add dataset characters
        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            self.stoi[char] = start_idx + i

        self.itos = {i: char for char, i in self.stoi.items()}

        # 3. Save to cache
        print(f"Saving vocabulary to {vocab_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

        with open(vocab_path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, indent=4)

        print(f"Vocabulary built. Size: {len(self.stoi)}")

    def __len__(self):
        return len(self.stoi)

    def text_to_sequence(self, text):
        """
        Converts an InChI string to a padded sequence of integers.

        Args:
            text (str): The InChI string.

        Returns:
            torch.Tensor: Tensor of shape (max_len,) containing integer indices.
        """
        # Start with SOS
        sequence = [self.sos_token_id]

        # Map characters
        for char in text:
            sequence.append(self.stoi.get(char, self.unk_token_id))

        # Add EOS
        sequence.append(self.eos_token_id)

        # Truncate if necessary (leaving space for EOS if strictly enforcing MAX_LEN,
        # but usually MAX_LEN is sufficient to hold the longest string + tokens)
        if len(sequence) > Config.MAX_LEN:
            sequence = sequence[: Config.MAX_LEN - 1] + [self.eos_token_id]

        # Pad
        padding_length = Config.MAX_LEN - len(sequence)
        if padding_length > 0:
            sequence.extend([self.pad_token_id] * padding_length)

        return torch.tensor(sequence, dtype=torch.long)

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of integers back to an InChI string.

        Args:
            sequence (list or torch.Tensor): Sequence of integer indices.

        Returns:
            str: The decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        result = []
        for idx in sequence:
            idx = int(idx)

            # Stop at EOS
            if idx == self.eos_token_id:
                break

            # Skip special tokens (SOS, PAD, UNK) in the output string
            if idx in [self.sos_token_id, self.pad_token_id, self.unk_token_id]:
                continue

            result.append(self.itos.get(idx, ""))

        return "".join(result)
