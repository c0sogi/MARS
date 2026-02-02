import os
import pandas as pd
import numpy as np
import torch
from library.config import Config


class InChiTokenizer:
    """
    Tokenizer for InChI strings.
    Handles character-level tokenization, special tokens, and vocabulary management.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"  # Index 0, also acts as CTC blank
    SOS_TOKEN = "<sos>"  # Index 1
    EOS_TOKEN = "<eos>"  # Index 2
    UNK_TOKEN = "<unk>"  # Index 3

    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, config: Config, load_cached_data: bool = True):
        self.config = config
        self.vocab_path = config.vocab_path
        self.train_metadata_path = config.train_metadata_path

        # Initialize vocabulary containers
        self.itos = {}  # int to string
        self.stoi = {}  # string to int

        # Ensure working directory exists
        os.makedirs(config.working_dir, exist_ok=True)

        # Load or build vocabulary
        if load_cached_data and os.path.exists(self.vocab_path):
            print(f"Loading vocabulary from {self.vocab_path}")
            self._load_vocab()
        else:
            print(f"Building vocabulary from {self.train_metadata_path}")
            self._build_vocab()
            self._save_vocab()

    def _build_vocab(self):
        """
        Builds vocabulary from the training metadata.
        """
        if not os.path.exists(self.train_metadata_path):
            raise FileNotFoundError(
                f"Training metadata not found at {self.train_metadata_path}"
            )

        df = pd.read_csv(self.train_metadata_path)

        # Extract all unique characters from InChI strings
        unique_chars = set()
        # We assume the 'InChI' column exists based on metadata generation script
        if "InChI" not in df.columns:
            raise ValueError("Metadata file does not contain 'InChI' column")

        for inchi in df["InChI"].astype(str):
            unique_chars.update(inchi)

        # Sort characters for deterministic behavior
        sorted_chars = sorted(list(unique_chars))

        # Create vocabulary list: special tokens + data characters
        vocab_list = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ] + sorted_chars

        # Create mappings
        self.itos = {i: char for i, char in enumerate(vocab_list)}
        self.stoi = {char: i for i, char in enumerate(vocab_list)}

        print(f"Vocabulary built. Size: {len(self.itos)}")

    def _save_vocab(self):
        """
        Saves the vocabulary list to a .npy file.
        """
        # Convert dictionary to list for saving
        vocab_list = [self.itos[i] for i in range(len(self.itos))]
        np.save(self.vocab_path, np.array(vocab_list))
        print(f"Vocabulary saved to {self.vocab_path}")

    def _load_vocab(self):
        """
        Loads the vocabulary list from a .npy file.
        """
        try:
            vocab_array = np.load(self.vocab_path)
            vocab_list = vocab_array.tolist()

            self.itos = {i: char for i, char in enumerate(vocab_list)}
            self.stoi = {char: i for i, char in enumerate(vocab_list)}
            print(f"Vocabulary loaded. Size: {len(self.itos)}")
        except Exception as e:
            print(f"Failed to load vocabulary: {e}. Rebuilding...")
            self._build_vocab()
            self._save_vocab()

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        """
        Converts a string to a tensor of indices.
        Adds SOS at the beginning and EOS at the end.

        Args:
            text (str): The InChI string.

        Returns:
            torch.Tensor: 1D tensor of indices (Long).
        """
        tokens = [self.SOS_ID]
        for char in text:
            tokens.append(self.stoi.get(char, self.UNK_ID))
        tokens.append(self.EOS_ID)

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """
        Converts a tensor of indices back to a string.
        Stops at EOS. Ignores PAD, SOS, UNK (optional, usually we keep UNK or replace it).

        Args:
            indices (torch.Tensor or list): Sequence of indices.

        Returns:
            str: Decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if idx == self.EOS_ID:
                break
            if idx == self.PAD_ID:
                continue
            if idx == self.SOS_ID:
                continue

            # Retrieve character
            char = self.itos.get(idx, "")
            chars.append(char)

        return "".join(chars)

    def get_pad_id(self):
        return self.PAD_ID

    def get_sos_id(self):
        return self.SOS_ID

    def get_eos_id(self):
        return self.EOS_ID

    def get_unk_id(self):
        return self.UNK_ID
