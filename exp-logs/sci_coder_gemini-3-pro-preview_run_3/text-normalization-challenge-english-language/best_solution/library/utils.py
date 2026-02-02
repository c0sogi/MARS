import os
import json
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


class CharTokenizer:
    """
    Character-level tokenizer for Seq2Seq models.
    Handles vocabulary building, encoding, and decoding with support for special tokens.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        # Initialize with special tokens defined in Config
        self.special_tokens = {
            Config.PAD_TOKEN: Config.PAD_IDX,
            Config.SOS_TOKEN: Config.SOS_IDX,
            Config.EOS_TOKEN: Config.EOS_IDX,
            Config.UNK_TOKEN: Config.UNK_IDX,
            Config.SEP_TOKEN: Config.SEP_IDX,
        }

        # Load special tokens into maps
        self.char2idx = self.special_tokens.copy()
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def build_vocab(self, texts, load_cached_data=True):
        """
        Builds vocabulary from a list of texts or loads from cache.

        Args:
            texts (list): List of strings to build vocabulary from.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        cache_path = os.path.join(Config.WORKING_DIR, "vocab.json")

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading vocab from {cache_path}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.char2idx = data["char2idx"]
                    # JSON keys are always strings, convert values back to int keys for idx2char
                    self.idx2char = {int(v): k for k, v in self.char2idx.items()}
                print(f"Loaded vocabulary. Size: {len(self.char2idx)}")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding vocabulary.")

        # 2. Build from scratch
        print("Building vocabulary from scratch...")
        unique_chars = set()

        # Efficiently collect all unique characters
        for text in texts:
            unique_chars.update(str(text))

        # Sort to ensure deterministic index assignment
        sorted_chars = sorted(list(unique_chars))

        # Determine starting index (avoid overwriting special tokens)
        # We assume special tokens occupy indices 0 to N
        start_idx = max(self.special_tokens.values()) + 1

        current_idx = start_idx
        for char in sorted_chars:
            if char not in self.char2idx:
                self.char2idx[char] = current_idx
                self.idx2char[current_idx] = char
                current_idx += 1

        print(f"Vocabulary built. Size: {len(self.char2idx)}")

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            # Save only char2idx, idx2char can be reconstructed
            json.dump({"char2idx": self.char2idx}, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {cache_path}")

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of token indices.

        Args:
            text (str): Input string.
            add_special_tokens (bool): If True, wraps the sequence with SOS and EOS tokens.

        Returns:
            list: List of integer indices.
        """
        indices = []
        # Convert text to string to handle potential non-string inputs gracefully
        for char in str(text):
            indices.append(self.char2idx.get(char, Config.UNK_IDX))

        if add_special_tokens:
            indices = [Config.SOS_IDX] + indices + [Config.EOS_IDX]

        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a list or tensor of indices back to a string.

        Args:
            indices (list or torch.Tensor): Input sequence of indices.
            remove_special_tokens (bool): If True, removes SOS, EOS, PAD, and SEP tokens.

        Returns:
            str: Decoded string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if remove_special_tokens:
                # Stop decoding if EOS is encountered
                if idx == Config.EOS_IDX:
                    break
                # Skip other special tokens
                if idx in [Config.SOS_IDX, Config.PAD_IDX, Config.SEP_IDX]:
                    continue

            char = self.idx2char.get(idx, Config.UNK_TOKEN)
            chars.append(char)

        return "".join(chars)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.char2idx)
