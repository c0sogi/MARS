import os
import json
import pandas as pd
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for InChI strings.
    Handles vocabulary building, encoding to tensor, and decoding to string.
    """

    def __init__(self, load_cached_data=True):
        self.stoi = {}
        self.itos = {}

        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # Reserved indices
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self._load_or_build_vocab(load_cached_data)

    def _load_or_build_vocab(self, load_cached_data):
        """
        Loads vocabulary from cache or builds it from training metadata.
        """
        vocab_path = Config.VOCAB_PATH

        # Logic: If load_cached_data is True AND file exists, load it.
        # Otherwise, build from scratch and save.
        if load_cached_data and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            try:
                with open(vocab_path, "r") as f:
                    self.stoi = json.load(f)
                # Reconstruct itos (keys in json are strings, convert back to int)
                self.itos = {int(v): k for k, v in self.stoi.items()}
            except Exception as e:
                print(f"Failed to load vocabulary: {e}. Rebuilding...")
                self._build_and_save_vocab(vocab_path)
        else:
            self._build_and_save_vocab(vocab_path)

        print(f"Vocabulary size: {len(self.stoi)}")

    def _build_and_save_vocab(self, vocab_path):
        """
        Builds vocabulary from training metadata and saves to JSON.
        """
        print("Building vocabulary from training metadata...")

        if not os.path.exists(Config.TRAIN_METADATA):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_METADATA}"
            )

        df = pd.read_csv(Config.TRAIN_METADATA)

        # Extract all unique characters from the InChI column
        # Using a set for efficiency
        text_blob = "".join(df["InChI"].dropna().astype(str).values)
        unique_chars = sorted(list(set(text_blob)))

        # Initialize with special tokens
        self.stoi = {
            self.pad_token: self.PAD_IDX,
            self.sos_token: self.SOS_IDX,
            self.eos_token: self.EOS_IDX,
            self.unk_token: self.UNK_IDX,
        }

        # Add characters to vocabulary
        # Start index after special tokens
        start_idx = 4
        for i, char in enumerate(unique_chars):
            self.stoi[char] = start_idx + i

        # Create inverse mapping
        self.itos = {v: k for k, v in self.stoi.items()}

        # Save to cache
        print(f"Saving vocabulary to {vocab_path}")
        with open(vocab_path, "w") as f:
            json.dump(self.stoi, f, indent=4)

    def __len__(self):
        return len(self.stoi)

    def encode(self, text):
        """
        Encodes a text string into a fixed-length tensor.
        Adds SOS and EOS tokens and pads to Config.MAX_LENGTH.

        Args:
            text (str): InChI string.

        Returns:
            torch.Tensor: LongTensor of shape (MAX_LENGTH,)
        """
        # Convert characters to indices
        indices = [self.stoi.get(char, self.UNK_IDX) for char in text]

        # Add SOS and EOS
        indices = [self.SOS_IDX] + indices + [self.EOS_IDX]

        # Handle Padding / Truncation
        max_len = Config.MAX_LENGTH
        current_len = len(indices)

        if current_len < max_len:
            # Pad
            padding = [self.PAD_IDX] * (max_len - current_len)
            indices.extend(padding)
        else:
            # Truncate (simple truncation, ensuring length is max_len)
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """
        Decodes a sequence of indices back into a string.
        Stops decoding when EOS token is encountered.

        Args:
            indices (list or torch.Tensor): Sequence of integer indices.

        Returns:
            str: Decoded InChI string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        decoded_chars = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx == self.PAD_IDX or idx == self.SOS_IDX:
                continue

            # Retrieve character, default to empty if not found
            char = self.itos.get(idx, "")
            decoded_chars.append(char)

        return "".join(decoded_chars)
