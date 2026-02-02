import os
import numpy as np
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, caching, padding, and special tokens.
    """

    def __init__(self, config: Config):
        self.config = config

        # Special tokens
        self.sos_token = config.sos_token
        self.eos_token = config.eos_token
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token

        # Fixed IDs for special tokens
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # Mappings
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts, load_cached_data=True):
        """
        Builds the vocabulary from a list of texts or loads it from cache.

        Args:
            texts (list or pd.Series): List of InChI strings.
            load_cached_data (bool): Whether to try loading from the cache file.
        """
        cache_path = self.config.tokenizer_cache_path
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        vocab_chars = None
        loaded_from_cache = False

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading tokenizer vocabulary from {cache_path}...")
                vocab_chars = np.load(cache_path)
                loaded_from_cache = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing vocabulary.")

        # 2. Compute from scratch if needed
        if not loaded_from_cache:
            print("Computing vocabulary from texts...")
            unique_chars = set()
            for text in texts:
                unique_chars.update(str(text))

            # Sort for determinism
            vocab_chars = np.array(sorted(list(unique_chars)))

            # Save to cache
            print(f"Saving tokenizer vocabulary to {cache_path}...")
            np.save(cache_path, vocab_chars)

        # 3. Build dictionaries
        # Initialize with special tokens
        self.char_to_int = {
            self.pad_token: self.pad_token_id,
            self.sos_token: self.sos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
        }

        # Add real characters starting from ID 4
        start_idx = 4
        for i, char in enumerate(vocab_chars):
            self.char_to_int[char] = start_idx + i

        self.int_to_char = {v: k for k, v in self.char_to_int.items()}
        self.vocab_size = len(self.char_to_int)

        print(f"Vocabulary size: {self.vocab_size}")

    def text_to_sequence(self, text):
        """
        Converts a text string to a padded sequence of integers.

        Args:
            text (str): Input InChI string.

        Returns:
            torch.LongTensor: Padded sequence tensor of shape (max_length,).
        """
        sequence = [self.sos_token_id]

        for char in str(text):
            if char in self.char_to_int:
                sequence.append(self.char_to_int[char])
            else:
                sequence.append(self.unk_token_id)

        sequence.append(self.eos_token_id)

        # Padding
        max_len = self.config.max_length
        if len(sequence) < max_len:
            sequence += [self.pad_token_id] * (max_len - len(sequence))
        else:
            # Truncate if necessary (though max_length should be sufficient based on EDA)
            sequence = sequence[: max_len - 1] + [self.eos_token_id]

        return torch.tensor(sequence, dtype=torch.long)

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of integers back to a text string.

        Args:
            sequence (list or torch.Tensor): List of token IDs.

        Returns:
            str: Decoded string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        chars = []
        for token_id in sequence:
            token_id = int(token_id)

            if token_id == self.sos_token_id:
                continue
            if token_id == self.eos_token_id:
                break
            if token_id == self.pad_token_id:
                break

            if token_id in self.int_to_char:
                chars.append(self.int_to_char[token_id])
            else:
                chars.append("?")  # Visual placeholder for unknown decoding issues

        return "".join(chars)
