import os
import pandas as pd
import numpy as np
import torch
from library.config import Config


class CharVocab:
    """
    Character-level Vocabulary for Text Normalization.
    Handles mapping between characters and integer indices.
    """

    def __init__(self):
        self.itos = {}
        self.stoi = {}

        # Initialize with special tokens from Config
        # Order is critical: PAD=0, SOS=1, EOS=2, UNK=3
        self.specials = [
            (Config.PAD_TOKEN, Config.PAD_IDX),
            (Config.SOS_TOKEN, Config.SOS_IDX),
            (Config.EOS_TOKEN, Config.EOS_IDX),
            (Config.UNK_TOKEN, Config.UNK_IDX),
        ]

        for token, idx in self.specials:
            self.itos[idx] = token
            self.stoi[token] = idx

    def __len__(self):
        return len(self.itos)

    def build_from_corpus(self, text_iterable):
        """
        Builds the vocabulary from an iterable of strings.
        Scans all characters, adds unique ones to the mapping.
        """
        unique_chars = set()
        for text in text_iterable:
            unique_chars.update(str(text))

        # Ensure the separator token is in the vocabulary
        unique_chars.add(Config.SEP_TOKEN)

        # Remove special tokens from the set if they appear in text
        # (They are already added with reserved indices)
        for token, _ in self.specials:
            if token in unique_chars:
                unique_chars.remove(token)

        # Sort characters for deterministic index assignment
        sorted_chars = sorted(list(unique_chars))

        # Add characters to vocab starting after the reserved indices
        start_idx = len(self.itos)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.itos[idx] = char
            self.stoi[char] = idx

    def encode(self, text, add_sos=False, add_eos=False):
        """
        Converts a string into a list of integer indices.
        """
        indices = []
        if add_sos:
            indices.append(self.stoi[Config.SOS_TOKEN])

        for char in str(text):
            # Map char to index, default to UNK if not found
            indices.append(self.stoi.get(char, self.stoi[Config.UNK_TOKEN]))

        if add_eos:
            indices.append(self.stoi[Config.EOS_TOKEN])

        return indices

    def decode(self, indices, remove_special=True):
        """
        Converts a list of indices (or tensor) back into a string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        tokens = []
        for idx in indices:
            if remove_special:
                # Skip Pad and SOS
                if idx in [Config.PAD_IDX, Config.SOS_IDX]:
                    continue
                # Stop at EOS
                if idx == Config.EOS_IDX:
                    break

            token = self.itos.get(idx, Config.UNK_TOKEN)
            tokens.append(token)

        return "".join(tokens)

    def save(self, path):
        """
        Saves the vocabulary to a .npy file.
        Saves the 'itos' mapping as a numpy array of strings.
        """
        # Create a list where index i contains the character for ID i
        vocab_list = [self.itos[i] for i in range(len(self.itos))]
        np.save(path, np.array(vocab_list))

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        # Load the array of characters
        vocab_array = np.load(path, allow_pickle=True)

        # Reconstruct mappings
        self.itos = {}
        self.stoi = {}

        for idx, token in enumerate(vocab_array):
            token = str(token)
            self.itos[idx] = token
            self.stoi[token] = idx


def get_vocab(load_cached_data=True):
    """
    Factory function to retrieve the vocabulary.
    Implements caching logic: Load if exists, else build and save.
    """
    vocab = CharVocab()

    # Ensure the cache directory exists
    os.makedirs(os.path.dirname(Config.VOCAB_CACHE), exist_ok=True)

    if load_cached_data and os.path.exists(Config.VOCAB_CACHE):
        print(f"Loading vocabulary from {Config.VOCAB_CACHE}")
        vocab.load(Config.VOCAB_CACHE)
    else:
        print("Building vocabulary from training data...")
        # Load training metadata
        # keep_default_na=False prevents "null" string from being NaN
        df = pd.read_csv(Config.TRAIN_FILE, keep_default_na=False)

        # Collect all text from 'before' and 'after' columns to cover all chars
        # Convert to string to ensure safety
        texts = pd.concat([df["before"], df["after"]]).astype(str).tolist()

        vocab.build_from_corpus(texts)

        print(f"Saving vocabulary to {Config.VOCAB_CACHE}")
        vocab.save(Config.VOCAB_CACHE)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab
