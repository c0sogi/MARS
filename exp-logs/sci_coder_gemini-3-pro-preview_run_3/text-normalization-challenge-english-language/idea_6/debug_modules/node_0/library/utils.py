import os
import random
import json
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CharTokenizer:
    """
    A character-level tokenizer that handles vocabulary creation,
    encoding, and decoding for the sequence-to-sequence model.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = {
            Config.PAD_TOKEN: Config.PAD_IDX,
            Config.SOS_TOKEN: Config.SOS_IDX,
            Config.EOS_TOKEN: Config.EOS_IDX,
            Config.UNK_TOKEN: Config.UNK_IDX,
            Config.SEP_TOKEN: Config.SEP_IDX,
        }
        # Initialize with special tokens
        self._reset_vocab()

    def _reset_vocab(self):
        """Resets vocabulary to only contain special tokens."""
        self.char2idx = self.special_tokens.copy()
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def fit_on_texts(self, texts):
        """
        Builds the vocabulary from a list of strings.

        Args:
            texts (iterable): An iterable of strings (e.g., list, pandas Series).
        """
        self._reset_vocab()
        unique_chars = set()

        for text in texts:
            if isinstance(text, str):
                unique_chars.update(text)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Start indexing after the last special token
        start_idx = max(self.special_tokens.values()) + 1

        for idx, char in enumerate(sorted_chars, start=start_idx):
            # Skip if char is already a special token (unlikely but safe)
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def save(self, path):
        """
        Saves the vocabulary to a JSON file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        Loads the vocabulary from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)

        # Reconstruct idx2char (keys in json are always strings, values are ints)
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}

    def encode(self, text, add_special_tokens=True):
        """
        Converts a string to a list of token indices.

        Args:
            text (str): Input string.
            add_special_tokens (bool): If True, adds SOS and EOS tokens.

        Returns:
            list: List of integer indices.
        """
        if not isinstance(text, str):
            text = str(text)

        indices = []
        for char in text:
            indices.append(self.char2idx.get(char, Config.UNK_IDX))

        if add_special_tokens:
            indices = [Config.SOS_IDX] + indices + [Config.EOS_IDX]

        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a list of token indices back to a string.

        Args:
            indices (list): List of integer indices.
            remove_special_tokens (bool): If True, removes SOS, EOS, PAD, UNK, SEP.

        Returns:
            str: Decoded string.
        """
        chars = []
        for idx in indices:
            # Handle tensor or numpy scalar inputs
            if hasattr(idx, "item"):
                idx = idx.item()

            # Stop decoding if EOS is encountered
            if remove_special_tokens and idx == Config.EOS_IDX:
                break

            char = self.idx2char.get(idx, Config.UNK_TOKEN)

            if remove_special_tokens:
                if idx in [Config.SOS_IDX, Config.PAD_IDX, Config.SEP_IDX]:
                    continue
                # Note: We usually keep UNK in output to indicate failure,
                # or we can skip it. Here we keep the token representation.

            chars.append(char)

        return "".join(chars)

    def __len__(self):
        return len(self.char2idx)
