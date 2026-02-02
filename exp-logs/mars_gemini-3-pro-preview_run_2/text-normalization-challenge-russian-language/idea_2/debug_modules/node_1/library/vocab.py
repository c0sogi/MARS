import os
import torch
import pandas as pd
from library.config import Config
from library.utils import save_data, load_data


class CharTokenizer:
    """
    A character-level tokenizer for text normalization tasks.
    Handles special tokens, encoding, decoding, and vocabulary management.
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

        # Initialize special tokens map based on Config
        # We assume the indices in Config are 0, 1, 2, 3, 4 respectively
        self.special_tokens = {
            Config.PAD_TOKEN: Config.PAD_IDX,
            Config.SOS_TOKEN: Config.SOS_IDX,
            Config.EOS_TOKEN: Config.EOS_IDX,
            Config.UNK_TOKEN: Config.UNK_IDX,
            Config.SEP_TOKEN: Config.SEP_IDX,
        }

        # Populate initial vocab with special tokens
        for token, idx in self.special_tokens.items():
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def fit(self, texts):
        """
        Builds the vocabulary from a list of strings.

        Args:
            texts (list[str]): List of strings to learn from.
        """
        unique_chars = set()
        for text in texts:
            # Ensure text is string and handle potential NaNs from raw data loading
            if pd.isna(text):
                continue
            unique_chars.update(str(text))

        # Sort characters to ensure deterministic vocabulary construction
        sorted_chars = sorted(list(unique_chars))

        # Start indexing after the last special token
        # We assume special tokens occupy the first N indices
        current_idx = max(self.token2idx.values()) + 1

        for char in sorted_chars:
            if char not in self.token2idx:
                self.token2idx[char] = current_idx
                self.idx2token[current_idx] = char
                current_idx += 1

    def encode(self, text, max_len=None, add_special_tokens=True):
        """
        Converts a string to a tensor of token indices.

        Args:
            text (str): The input string.
            max_len (int, optional): Maximum sequence length for truncation/padding.
            add_special_tokens (bool): If True, wraps sequence with SOS and EOS.

        Returns:
            torch.Tensor: A LongTensor of indices.
        """
        text = str(text)
        indices = []

        if add_special_tokens:
            indices.append(Config.SOS_IDX)

        for char in text:
            indices.append(self.token2idx.get(char, Config.UNK_IDX))

        if add_special_tokens:
            indices.append(Config.EOS_IDX)

        # Handle Truncation
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
                # If we truncated and were supposed to have special tokens,
                # strictly speaking we just cut off.
                # Optionally enforce EOS at end, but simple truncation is standard.

            # Handle Padding
            if len(indices) < max_len:
                padding = [Config.PAD_IDX] * (max_len - len(indices))
                indices.extend(padding)

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a sequence of indices back to a string.

        Args:
            indices (list or torch.Tensor): The indices to decode.
            remove_special_tokens (bool): If True, omits PAD, SOS, EOS, UNK, SEP.

        Returns:
            str: The decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            # Always stop at EOS
            if idx == Config.EOS_IDX:
                break

            if remove_special_tokens:
                if idx in [
                    Config.PAD_IDX,
                    Config.SOS_IDX,
                    Config.UNK_IDX,
                    Config.SEP_IDX,
                ]:
                    continue

            # Retrieve character, ignore if somehow invalid
            char = self.idx2token.get(idx, "")
            chars.append(char)

        return "".join(chars)

    def to_dict(self):
        """Returns the vocabulary dictionary for serialization."""
        return self.token2idx

    def from_dict(self, token2idx):
        """Restores vocabulary from a dictionary."""
        self.token2idx = token2idx
        # JSON keys are strings, but our indices must be integers in idx2token
        self.idx2token = {int(v): k for k, v in token2idx.items()}


def get_tokenizer(load_cached_data=True):
    """
    Retrieves the CharTokenizer.

    Logic:
    1. If load_cached_data is True and file exists: Load from JSON.
    2. Else: Compute from training data, save to JSON, and return.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        CharTokenizer: The initialized tokenizer.
    """
    vocab_path = Config.VOCAB_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(vocab_path):
        try:
            vocab_dict = load_data(vocab_path)
            tokenizer = CharTokenizer()
            tokenizer.from_dict(vocab_dict)
            return tokenizer
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute from scratch
    # Ensure working directory exists (handled by Config.setup_directories usually,
    # but we ensure path existence via save_data internal logic or manual check)

    # Load training data
    # We read the 'before' and 'after' columns to capture all possible characters
    # including those in the target transliterations.
    df = pd.read_csv(Config.TRAIN_DATA_PATH)

    # Collect all unique text
    # Using set for efficiency
    texts = set(df["before"].dropna().astype(str).unique())
    texts.update(df["after"].dropna().astype(str).unique())

    # Initialize and fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(list(texts))

    # 3. Save to cache
    save_data(tokenizer.to_dict(), vocab_path)

    return tokenizer
