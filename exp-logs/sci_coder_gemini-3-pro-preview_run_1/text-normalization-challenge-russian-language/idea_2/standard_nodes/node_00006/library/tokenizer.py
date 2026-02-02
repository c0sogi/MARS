import os
import json
import pandas as pd
import torch
from library.config import Config


class CharTokenizer:
    """
    Character-level tokenizer for the Hybrid Cascade model.
    Handles vocabulary creation, encoding, and decoding of text sequences.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        # Special tokens from Config
        self.pad_token = Config.PAD_TOKEN
        self.sos_token = Config.SOS_TOKEN
        self.eos_token = Config.EOS_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initializes the vocabulary with special tokens."""
        special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.char2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2char = {idx: token for idx, token in enumerate(special_tokens)}

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def pad_token_id(self):
        return self.char2idx[self.pad_token]

    @property
    def sos_token_id(self):
        return self.char2idx[self.sos_token]

    @property
    def eos_token_id(self):
        return self.char2idx[self.eos_token]

    @property
    def unk_token_id(self):
        return self.char2idx[self.unk_token]

    def fit(self, texts):
        """
        Builds vocabulary from a list of text strings.

        Args:
            texts (list): List of strings to learn from.
        """
        unique_chars = set()
        for text in texts:
            if pd.isna(text):
                continue
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Add to vocabulary (start after special tokens)
        start_idx = len(self.char2idx)
        for i, char in enumerate(sorted_chars):
            self.char2idx[char] = start_idx + i
            self.idx2char[start_idx + i] = char

    def encode(self, text, max_len=None, add_special_tokens=True, return_tensor=True):
        """
        Converts a string to a sequence of token IDs.

        Args:
            text (str): Input string.
            max_len (int, optional): Maximum sequence length. Truncates if longer.
            add_special_tokens (bool): If True, adds SOS and EOS tokens.
            return_tensor (bool): If True, returns a torch.LongTensor.

        Returns:
            list or torch.Tensor: Sequence of token IDs.
        """
        if pd.isna(text):
            text = ""
        text = str(text)

        indices = [self.char2idx.get(char, self.unk_token_id) for char in text]

        if add_special_tokens:
            indices = [self.sos_token_id] + indices + [self.eos_token_id]

        if max_len is not None:
            # Truncate if necessary (keeping EOS if present is usually preferred,
            # but simple truncation is standard for fixed buffers)
            if len(indices) > max_len:
                indices = indices[:max_len]
                # Ensure EOS is at the end if we added special tokens and truncated
                if add_special_tokens:
                    indices[-1] = self.eos_token_id

            # Pad if necessary
            if len(indices) < max_len:
                indices = indices + [self.pad_token_id] * (max_len - len(indices))

        if return_tensor:
            return torch.tensor(indices, dtype=torch.long)
        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a sequence of token IDs back to a string.

        Args:
            indices (list or torch.Tensor): Sequence of token IDs.
            remove_special_tokens (bool): If True, removes PAD, SOS, EOS.

        Returns:
            str: Decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            token = self.idx2char.get(idx, self.unk_token)

            if remove_special_tokens:
                if token in [self.pad_token, self.sos_token, self.eos_token]:
                    continue

            chars.append(token)

        return "".join(chars)

    def save(self, path):
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer vocabulary saved to {path}")

    def load(self, path):
        """Loads the vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)

        # Reconstruct idx2char
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        print(f"Tokenizer vocabulary loaded from {path} (Size: {len(self.char2idx)})")

    def build_vocab(self, data_path, load_cached=True):
        """
        Builds the vocabulary from the dataset or loads from cache.
        Strictly follows the caching logic requirement.

        Args:
            data_path (str): Path to the training CSV file.
            load_cached (bool): Whether to attempt loading from cache.
        """
        cache_path = Config.TOKENIZER_PATH

        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached and os.path.exists(cache_path):
            try:
                self.load(cache_path)
                return
            except Exception as e:
                print(f"Failed to load cached tokenizer: {e}. Rebuilding...")

        # 2. IF loading fails OR load_cached_data is False:
        print("Building tokenizer vocabulary from scratch...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Read data
        # We need both 'before' and 'after' columns to capture all characters
        # Using usecols to save memory
        try:
            df = pd.read_csv(data_path, usecols=["before", "after"], dtype=str)
        except ValueError:
            # Fallback if columns might be named differently or file is empty
            df = pd.read_csv(data_path, dtype=str)

        texts = df["before"].dropna().tolist() + df["after"].dropna().tolist()

        # Fit tokenizer
        self.fit(texts)

        # Save result
        self.save(cache_path)
