import os
import json
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer class for converting InChI strings to numerical sequences and vice versa.
    Handles vocabulary building, caching, and special token management.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the Tokenizer.

        Args:
            load_cached_data (bool): If True, attempts to load the vocabulary from cache.
                                     If False or load fails, rebuilds from training data.
        """
        self.stoi = {}
        self.itos = {}

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_path = Config.TOKENIZER_PATH

        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading tokenizer vocabulary from {cache_path}...")
                with open(cache_path, "r") as f:
                    self.stoi = json.load(f)
                # Reconstruct itos (keys in json are strings, values are ints)
                # We need itos to map int -> string
                self.itos = {int(v): k for k, v in self.stoi.items()}
                loaded = True
            except Exception as e:
                print(f"Failed to load cached tokenizer: {e}")
                loaded = False

        if not loaded:
            print("Building tokenizer vocabulary from training metadata...")
            self._build_vocab()
            print(f"Saving tokenizer vocabulary to {cache_path}...")
            with open(cache_path, "w") as f:
                json.dump(self.stoi, f, indent=4)

    def _build_vocab(self):
        """
        Builds the vocabulary from the training metadata.
        """
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
            )

        df = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # Extract unique characters from the InChI column
        # Using unique() on the column first significantly speeds up processing
        unique_labels = df["InChI"].unique()

        chars = set()
        for label in unique_labels:
            chars.update(label)

        # Sort characters to ensure deterministic vocabulary
        unique_chars = sorted(list(chars))

        # Define special tokens
        self.stoi = {
            Config.PAD_TOKEN: 0,
            Config.SOS_TOKEN: 1,
            Config.EOS_TOKEN: 2,
            Config.UNK_TOKEN: 3,
        }

        # Add dataset characters to vocabulary
        idx = 4
        for char in unique_chars:
            self.stoi[char] = idx
            idx += 1

        # Create inverse mapping
        self.itos = {v: k for k, v in self.stoi.items()}
        print(f"Vocabulary built. Size: {len(self.stoi)}")

    def __len__(self):
        """
        Returns the size of the vocabulary.
        """
        return len(self.stoi)

    def text_to_sequence(self, text, max_length=None, padding=True):
        """
        Converts an InChI string to a sequence of indices.

        Args:
            text (str): The InChI string.
            max_length (int): Max length for padding/truncation. Defaults to Config.MAX_SEQUENCE_LENGTH.
            padding (bool): Whether to pad the sequence.

        Returns:
            torch.Tensor: Tensor of indices (LongTensor).
        """
        if max_length is None:
            max_length = Config.MAX_SEQUENCE_LENGTH

        sequence = [self.stoi[Config.SOS_TOKEN]]

        for char in text:
            sequence.append(self.stoi.get(char, self.stoi[Config.UNK_TOKEN]))

        sequence.append(self.stoi[Config.EOS_TOKEN])

        # Truncate if necessary
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        # Pad
        if padding:
            pad_len = max_length - len(sequence)
            if pad_len > 0:
                sequence.extend([self.stoi[Config.PAD_TOKEN]] * pad_len)

        return torch.tensor(sequence, dtype=torch.long)

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of indices back to an InChI string.

        Args:
            sequence (list or torch.Tensor): Sequence of indices.

        Returns:
            str: The decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()

        result = []
        for idx in sequence:
            idx = int(idx)

            # Skip Start of Sequence
            if idx == self.stoi[Config.SOS_TOKEN]:
                continue

            # Stop at End of Sequence
            if idx == self.stoi[Config.EOS_TOKEN]:
                break

            # Skip Padding (if encountered before EOS)
            if idx == self.stoi[Config.PAD_TOKEN]:
                continue

            char = self.itos.get(idx, "")
            result.append(char)

        return "".join(result)
