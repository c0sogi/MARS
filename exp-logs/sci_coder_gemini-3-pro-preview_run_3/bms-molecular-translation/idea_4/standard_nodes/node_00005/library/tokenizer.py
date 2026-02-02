import os
import pandas as pd
import numpy as np
import torch
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to numerical sequences and vice versa.
    """

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        # Special tokens
        self.pad_token = Config.PAD_TOKEN
        self.sos_token = Config.SOS_TOKEN
        self.eos_token = Config.EOS_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # Order matters for indices: PAD=0, SOS=1, EOS=2, UNK=3
        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]

    def fit_on_texts(self, load_cached_data=True):
        """
        Builds the vocabulary from the training metadata.
        Implements strict caching mechanism using parquet.

        Args:
            load_cached_data (bool): Whether to try loading from cache.
        """
        cache_path = os.path.join(Config.WORKING_DIR, "vocab.parquet")

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        vocab_list = []

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading vocabulary from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                vocab_list = df["char"].tolist()
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding vocabulary.")
                vocab_list = []  # Ensure we fall through to rebuild

        # 2. Rebuild if needed
        if not vocab_list:
            print("Building vocabulary from training metadata...")
            if not os.path.exists(Config.TRAIN_METADATA_PATH):
                raise FileNotFoundError(
                    f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
                )

            train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            # Ensure strings
            texts = train_df["InChI"].astype(str).values

            # Extract unique characters
            unique_chars = set()
            for text in texts:
                unique_chars.update(text)

            # Sort for determinism
            sorted_chars = sorted(list(unique_chars))

            # Combine special tokens and extracted characters
            vocab_list = self.special_tokens + sorted_chars

            # Save to cache
            print(f"Saving vocabulary to cache: {cache_path}")
            df_vocab = pd.DataFrame({"char": vocab_list})
            df_vocab.to_parquet(cache_path, index=False)

        # 3. Create mappings
        self.idx_to_char = {i: char for i, char in enumerate(vocab_list)}
        self.char_to_idx = {char: i for i, char in enumerate(vocab_list)}

        print(f"Vocabulary built. Size: {len(self.char_to_idx)}")

    def text_to_sequence(self, text):
        """
        Converts an InChI string to a sequence of indices.

        Args:
            text (str): The InChI string.

        Returns:
            list[int]: The sequence of indices.
        """
        sequence = [self.char_to_idx[self.sos_token]]

        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx[self.unk_token])

        sequence.append(self.char_to_idx[self.eos_token])

        # Truncate if necessary
        if len(sequence) > Config.MAX_TEXT_LEN:
            sequence = sequence[: Config.MAX_TEXT_LEN]
            # Force EOS at the end if truncated
            sequence[-1] = self.char_to_idx[self.eos_token]

        # Pad
        if len(sequence) < Config.MAX_TEXT_LEN:
            pad_len = Config.MAX_TEXT_LEN - len(sequence)
            sequence += [self.char_to_idx[self.pad_token]] * pad_len

        return sequence

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of indices back to an InChI string.

        Args:
            sequence (list[int] or torch.Tensor): The sequence of indices.

        Returns:
            str: The decoded InChI string.
        """
        result = []
        for idx in sequence:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            # Stop at EOS
            if idx == self.char_to_idx[self.eos_token]:
                break

            # Skip PAD and SOS
            if (
                idx == self.char_to_idx[self.pad_token]
                or idx == self.char_to_idx[self.sos_token]
            ):
                continue

            if idx in self.idx_to_char:
                result.append(self.idx_to_char[idx])

        return "".join(result)

    def __len__(self):
        return len(self.char_to_idx)
