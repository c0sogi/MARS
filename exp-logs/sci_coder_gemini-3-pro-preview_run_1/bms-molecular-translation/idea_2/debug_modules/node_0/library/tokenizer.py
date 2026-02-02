import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, caching, and sequence padding.
    """

    def __init__(self, config: Config):
        self.config = config
        self.stoi = {}
        self.itos = {}

        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        # Reserved indices
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2

        self.vocab = []

    def load_or_build_vocab(self, load_cached_data: bool = True):
        """
        Loads the vocabulary from a cached .npy file or builds it from the training metadata.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
        """
        cache_path = self.config.tokenizer_path
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        loaded = False

        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading tokenizer vocabulary from {cache_path}...")
                self.vocab = np.load(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cached tokenizer: {e}. Rebuilding...")
                loaded = False

        if not loaded:
            print("Building tokenizer vocabulary from training metadata...")
            # Load training metadata
            if not os.path.exists(self.config.train_metadata_path):
                raise FileNotFoundError(
                    f"Training metadata not found at {self.config.train_metadata_path}"
                )

            df = pd.read_csv(self.config.train_metadata_path)

            # Extract all unique characters from InChI strings
            # We use a set for efficiency
            unique_chars = set()
            # To speed up, we can concatenate a sample or process in chunks if memory is tight,
            # but for 1.5M short strings, processing all is feasible.
            # Using a set comprehension over all strings
            all_inchis = df["InChI"].astype(str).tolist()
            for text in all_inchis:
                unique_chars.update(text)

            # Sort characters for determinism
            sorted_chars = sorted(list(unique_chars))

            # Create full vocabulary list: special tokens + sorted characters
            # Note: PAD, SOS, EOS are at 0, 1, 2
            self.vocab = np.array(
                [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] + sorted_chars
            )

            # Save to cache
            print(f"Saving tokenizer vocabulary to {cache_path}...")
            np.save(cache_path, self.vocab)

        # Build mapping dictionaries
        self.stoi = {char: idx for idx, char in enumerate(self.vocab)}
        self.itos = {idx: char for idx, char in enumerate(self.vocab)}

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Vocabulary: {self.vocab}")

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Converts an InChI string to a padded integer tensor.
        Adds SOS at the start and EOS at the end.

        Args:
            text (str): Input InChI string.

        Returns:
            torch.Tensor: Tensor of shape (max_len,) containing integer indices.
        """
        # Convert characters to indices
        # If a character is not in vocab, we could raise error or ignore.
        # Given the closed dataset, we assume all training chars cover test chars.
        sequence = [self.stoi[c] for c in text if c in self.stoi]

        # Add SOS and EOS
        sequence = [self.SOS_IDX] + sequence + [self.EOS_IDX]

        # Truncate if longer than max_len
        if len(sequence) > self.config.max_len:
            sequence = sequence[: self.config.max_len]
            # Ensure EOS is at the end if truncated (optional, but good practice for validity)
            sequence[-1] = self.EOS_IDX

        # Pad
        padding_length = self.config.max_len - len(sequence)
        if padding_length > 0:
            sequence = sequence + [self.PAD_IDX] * padding_length

        return torch.tensor(sequence, dtype=torch.long)

    def sequence_to_text(self, sequence: torch.Tensor) -> str:
        """
        Converts an integer tensor back to an InChI string.
        Stops at EOS token and ignores PAD/SOS tokens.

        Args:
            sequence (torch.Tensor or list): Sequence of integer indices.

        Returns:
            str: Decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        result = []
        for idx in sequence:
            idx = int(idx)

            if idx == self.SOS_IDX:
                continue
            if idx == self.EOS_IDX:
                break
            if idx == self.PAD_IDX:
                continue

            if idx in self.itos:
                result.append(self.itos[idx])

        return "".join(result)

    def __len__(self):
        return len(self.vocab)
