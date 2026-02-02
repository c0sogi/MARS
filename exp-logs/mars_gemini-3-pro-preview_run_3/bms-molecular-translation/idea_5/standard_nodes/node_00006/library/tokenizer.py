import os
import json
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, caching, and sequence padding.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build_vocab(self, load_cached_data: bool = True):
        """
        Builds the vocabulary from the training metadata or loads it from cache.

        Args:
            load_cached_data (bool): If True, attempts to load the vocabulary from
                                     Config.VOCAB_PATH. If False or loading fails,
                                     recomputes the vocabulary.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        vocab_path = Config.VOCAB_PATH
        loaded = False

        if load_cached_data and os.path.exists(vocab_path):
            try:
                print(f"Loading vocabulary from {vocab_path}...")
                with open(vocab_path, "r") as f:
                    self.stoi = json.load(f)
                loaded = True
            except Exception as e:
                print(f"Failed to load cached vocabulary: {e}")
                loaded = False

        if not loaded:
            print("Building vocabulary from training metadata...")
            # Load training metadata
            if not os.path.exists(Config.TRAIN_METADATA):
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA}"
                )

            df = pd.read_csv(Config.TRAIN_METADATA)

            # Collect all unique characters
            unique_chars = set()
            # We iterate over the column to ensure we catch everything.
            # InChI strings are relatively short, so this is feasible.
            # Using a set update with a joined string of chunks can be memory heavy,
            # so we iterate.
            for text in df["InChI"].dropna():
                unique_chars.update(text)

            sorted_chars = sorted(list(unique_chars))

            # Initialize stoi with special tokens
            self.stoi = {
                self.PAD_TOKEN: self.PAD_IDX,
                self.SOS_TOKEN: self.SOS_IDX,
                self.EOS_TOKEN: self.EOS_IDX,
            }

            # Add characters to stoi
            start_idx = 3
            for i, char in enumerate(sorted_chars):
                self.stoi[char] = start_idx + i

            # Save to cache
            print(f"Saving vocabulary to {vocab_path}...")
            with open(vocab_path, "w") as f:
                json.dump(self.stoi, f, indent=4)

        # Build itos (index to string)
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary built. Size: {self.vocab_size}")

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Converts an InChI string to a padded tensor sequence.

        Args:
            text (str): The InChI string.

        Returns:
            torch.Tensor: A LongTensor of shape (MAX_LEN,) containing indices.
        """
        # Convert text to indices
        # If a char is not in vocab, this will raise a KeyError.
        # Given the closed nature of InChI syntax and training set coverage, this is acceptable.
        indices = [self.stoi[char] for char in text]

        # Add SOS and EOS
        indices = [self.SOS_IDX] + indices + [self.EOS_IDX]

        # Pad or Truncate
        max_len = Config.MAX_LEN
        current_len = len(indices)

        if current_len < max_len:
            # Pad
            padding = [self.PAD_IDX] * (max_len - current_len)
            indices.extend(padding)
        else:
            # Truncate (keep SOS, truncate end, ensure EOS is at the very end if we strictly enforced structure,
            # but usually we just cut off. Ideally MAX_LEN is large enough).
            # Here we truncate and force the last token to be EOS if we cut off real data,
            # but standard practice for fixed tensor size is just simple truncation.
            indices = indices[:max_len]
            # Optional: Force EOS at end if truncated?
            # indices[-1] = self.EOS_IDX
            # For this task, MAX_LEN=450 covers the dataset (max ~403), so truncation shouldn't happen.

        return torch.tensor(indices, dtype=torch.long)

    def sequence_to_text(self, sequence: torch.Tensor) -> str:
        """
        Converts a sequence of indices back to a string.

        Args:
            sequence (torch.Tensor or list): A sequence of indices.

        Returns:
            str: The reconstructed InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()

        result = []
        for idx in sequence:
            idx = int(idx)

            if idx == self.EOS_IDX:
                break

            if idx == self.SOS_IDX or idx == self.PAD_IDX:
                continue

            if idx in self.itos:
                result.append(self.itos[idx])

        return "".join(result)

    def get_vocab_size(self) -> int:
        return self.vocab_size
