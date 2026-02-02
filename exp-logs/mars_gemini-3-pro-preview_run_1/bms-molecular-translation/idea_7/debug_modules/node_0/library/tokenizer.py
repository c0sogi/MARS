import os
import json
import numpy as np
import pandas as pd
from library.config import Config


class Tokenizer:
    """
    Character-level tokenizer for InChI strings.
    Handles vocabulary building, encoding, decoding, and persistence.
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        # Initial special tokens list
        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]

    def fit_on_texts(self, texts=None, load_cached_data=True):
        """
        Builds the vocabulary from texts or loads it from cache.

        Args:
            texts (list): List of InChI strings. If None, tries to load from Config.TRAIN_METADATA.
            load_cached_data (bool): Whether to try loading from the cached JSON file.
        """
        cache_path = Config.TOKENIZER_PATH
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading tokenizer vocabulary from {cache_path}")
                self.load(cache_path)
                return
            except Exception as e:
                print(f"Failed to load tokenizer cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Fitting tokenizer on training data...")

        if texts is None:
            if os.path.exists(Config.TRAIN_METADATA):
                print(f"Loading texts from {Config.TRAIN_METADATA}")
                df = pd.read_csv(Config.TRAIN_METADATA)
                texts = df["InChI"].astype(str).tolist()
            else:
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA} and no texts provided."
                )

        # Extract unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Sort characters for deterministic ordering
        sorted_chars = sorted(list(unique_chars))

        # Build mappings
        self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)

        for idx, char in enumerate(sorted_chars):
            self.token2id[char] = start_idx + idx

        self.id2token = {v: k for k, v in self.token2id.items()}

        print(f"Vocabulary size: {len(self.token2id)}")

        # 3. Save to cache
        print(f"Saving tokenizer vocabulary to {cache_path}")
        self.save(cache_path)

    def encode(self, text):
        """
        Encodes a text string into a padded sequence of integers.

        Args:
            text (str): Input InChI string.

        Returns:
            np.ndarray: Array of integers of shape (MAX_LEN,).
        """
        text = str(text)
        sequence = [self.token2id[self.sos_token]]

        for char in text:
            # Use UNK token if character is not in vocabulary
            sequence.append(self.token2id.get(char, self.token2id[self.unk_token]))

        sequence.append(self.token2id[self.eos_token])

        # Pad or truncate to MAX_LEN
        max_len = Config.MAX_LEN
        if len(sequence) < max_len:
            # Pad
            padding = [self.token2id[self.pad_token]] * (max_len - len(sequence))
            sequence.extend(padding)
        else:
            # Truncate (keep SOS, truncate tail, ensure EOS at end if desired,
            # but standard truncation usually just cuts off.
            # Here we strictly cut to MAX_LEN)
            sequence = sequence[:max_len]

        return np.array(sequence, dtype=np.int64)

    def decode(self, sequence):
        """
        Decodes a sequence of integers back into a string.
        Stops decoding at EOS token and ignores PAD/SOS/UNK tokens as appropriate.

        Args:
            sequence (list or np.ndarray): Sequence of token IDs.

        Returns:
            str: Decoded string.
        """
        decoded_chars = []

        for token_id in sequence:
            token_id = int(token_id)

            if token_id == self.token2id[self.eos_token]:
                break

            if token_id == self.token2id[self.pad_token]:
                continue

            if token_id == self.token2id[self.sos_token]:
                continue

            # Retrieve token, default to empty if not found (shouldn't happen)
            token = self.id2token.get(token_id, "")
            decoded_chars.append(token)

        return "".join(decoded_chars)

    def save(self, path):
        """
        Saves the token2id mapping to a JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.token2id, f, indent=4)

    def load(self, path):
        """
        Loads the token2id mapping from a JSON file and reconstructs id2token.
        """
        with open(path, "r") as f:
            self.token2id = json.load(f)
        # Reconstruct id2token (keys in JSON are strings, need to be ints for decoding)
        self.id2token = {int(v): k for k, v in self.token2id.items()}

    def __len__(self):
        return len(self.token2id)
