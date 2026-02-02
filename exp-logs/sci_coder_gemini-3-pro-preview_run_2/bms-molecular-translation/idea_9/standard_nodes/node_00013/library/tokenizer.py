import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class InChiTokenizer:
    def __init__(self, load_cached_data=True):
        """
        Initializes the tokenizer.

        Args:
            load_cached_data (bool): Whether to try loading the vocabulary from cache.
        """
        self.special_tokens = Config.SPECIAL_TOKENS
        self.vocab = self.load_vocab(load_cached_data=load_cached_data)

        # Create mappings
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx2token = {i: t for i, t in enumerate(self.vocab)}

        # Cache special token indices for quick access
        self.blk_idx = self.token2idx[Config.CTC_BLANK]
        self.pad_idx = self.token2idx[Config.PAD_TOKEN]
        self.sos_idx = self.token2idx[Config.SOS_TOKEN]
        self.eos_idx = self.token2idx[Config.EOS_TOKEN]
        self.unk_idx = self.token2idx[Config.UNK_TOKEN]

    def load_vocab(self, load_cached_data: bool) -> list:
        """
        Loads the vocabulary from cache or builds it from the training metadata.
        Strictly follows the caching mechanism requirements.

        Args:
            load_cached_data (bool): If True, attempts to load from file.

        Returns:
            list: The vocabulary list.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(Config.VOCAB_CACHE_PATH), exist_ok=True)

        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(Config.VOCAB_CACHE_PATH):
            try:
                vocab = np.load(Config.VOCAB_CACHE_PATH).tolist()
                # print(f"Loaded vocabulary from cache: {Config.VOCAB_CACHE_PATH}")
                return vocab
            except Exception:
                # print("Failed to load cached vocabulary. Rebuilding...")
                pass

        # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
        # print("Building vocabulary from training metadata...")

        if not os.path.exists(Config.TRAIN_METADATA):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_METADATA}"
            )

        df = pd.read_csv(Config.TRAIN_METADATA)

        # Extract all unique characters from the InChI column
        unique_chars = set()
        # We assume InChI column exists and is of string type
        # Using a set comprehension over all strings
        for text in df["InChI"].dropna().astype(str):
            unique_chars.update(text)

        # Sort characters for determinism
        sorted_chars = sorted(list(unique_chars))

        # Prepend special tokens defined in Config
        # Order: [Specials, Characters]
        full_vocab = self.special_tokens + sorted_chars

        # Save the result to the cache directory
        np.save(Config.VOCAB_CACHE_PATH, np.array(full_vocab))
        # print(f"Vocabulary saved to {Config.VOCAB_CACHE_PATH}")

        return full_vocab

    def text_to_sequence(self, text, add_sos=False, add_eos=False):
        """
        Converts a text string to a sequence of indices.

        Args:
            text (str): The input InChI string.
            add_sos (bool): Whether to prepend the Start-Of-Sequence token.
            add_eos (bool): Whether to append the End-Of-Sequence token.

        Returns:
            list[int]: The sequence of token indices.
        """
        sequence = []
        if add_sos:
            sequence.append(self.sos_idx)

        for char in text:
            sequence.append(self.token2idx.get(char, self.unk_idx))

        if add_eos:
            sequence.append(self.eos_idx)

        return sequence

    def sequence_to_text(self, sequence, remove_special=True):
        """
        Converts a sequence of indices back to a text string.
        Stops decoding if EOS token is encountered.

        Args:
            sequence (list[int] or torch.Tensor): The sequence of indices.
            remove_special (bool): If True, filters out special tokens from the output string.

        Returns:
            str: The decoded string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        chars = []
        for idx in sequence:
            idx = int(idx)

            # Stop at EOS
            if idx == self.eos_idx:
                break

            # Skip padding
            if idx == self.pad_idx:
                continue

            token = self.idx2token.get(idx, Config.UNK_TOKEN)

            if remove_special and token in self.special_tokens:
                continue

            chars.append(token)

        return "".join(chars)

    def decode_ctc_greedy(self, sequence):
        """
        Decodes a sequence of indices using CTC greedy decoding logic.
        1. Collapse repeated consecutive indices.
        2. Remove blank indices.

        Args:
            sequence (list[int] or torch.Tensor): The raw sequence of indices (usually from argmax).

        Returns:
            str: The decoded string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        collapsed_indices = []
        prev_idx = -1

        for idx in sequence:
            idx = int(idx)
            if idx != prev_idx:
                collapsed_indices.append(idx)
                prev_idx = idx

        # Remove blanks and convert to chars
        chars = []
        for idx in collapsed_indices:
            if idx != self.blk_idx:
                chars.append(self.idx2token.get(idx, Config.UNK_TOKEN))

        return "".join(chars)

    def pad_sequence(self, sequence, max_len, padding_value=None):
        """
        Pads a sequence to the specified maximum length.

        Args:
            sequence (list[int]): The input sequence.
            max_len (int): The target length.
            padding_value (int, optional): The value to pad with. Defaults to PAD_IDX.

        Returns:
            list[int]: The padded sequence.
        """
        if padding_value is None:
            padding_value = self.pad_idx

        if len(sequence) >= max_len:
            return sequence[:max_len]

        return sequence + [padding_value] * (max_len - len(sequence))

    def get_vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)
