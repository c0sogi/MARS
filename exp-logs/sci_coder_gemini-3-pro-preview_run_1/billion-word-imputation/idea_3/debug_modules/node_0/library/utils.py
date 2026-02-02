import os
import collections
import numpy as np
import pandas as pd
from library.config import Config

# Define special tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


def tokenize(text):
    """
    Splits text into tokens based on whitespace.
    The dataset is already pre-tokenized with spaces around punctuation.
    """
    if not isinstance(text, str):
        return []
    return text.strip().split()


class Vocabulary:
    def __init__(self):
        self.stoi = {}
        self.itos = []

    def __len__(self):
        return len(self.itos)

    def build(self, sentences, max_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ):
        """
        Builds vocabulary from a list of sentences.
        """
        counter = collections.Counter()
        for sentence in sentences:
            counter.update(tokenize(sentence))

        # Define special tokens
        # Order: PAD=0, UNK=1, SOS=2, EOS=3, MASK=4 (if used)
        special_tokens = [
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            SOS_TOKEN,
            EOS_TOKEN,
            Config.MASK_TOKEN,
        ]

        # Start with special tokens
        self.itos = list(special_tokens)

        # Add most frequent words
        # We subtract len(special_tokens) from max_size to ensure total size <= max_size
        target_vocab_count = max_size - len(special_tokens)

        # Filter by min_freq and sort by frequency
        valid_tokens = [w for w, c in counter.most_common() if c >= min_freq]

        # Truncate
        self.itos.extend(valid_tokens[:target_vocab_count])

        # Build stoi
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        print(f"Vocabulary built. Size: {len(self.itos)}")
        print(f"Top 5 frequent words: {valid_tokens[:5]}")

    def save(self, path):
        """
        Saves the vocabulary (itos list) to a .npy file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.itos))
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        # Load numpy array and convert to list
        # We use allow_pickle=True because loading an array of strings/unicode often requires it internally
        # or we rely on default behavior. To be strictly safe against the "No pickle" rule for data *transfer*,
        # we note that numpy's native format is binary. However, if strict no-pickle is enforced for security,
        # we can assume standard numpy save/load for string arrays is acceptable as per standard ML practices
        # unless strictly prohibited. The prompt says "Do NOT use pickle", but "Use ... npy".
        # np.load of a string array is the standard way to use npy.
        vocab_array = np.load(path)
        self.itos = vocab_array.tolist()
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        print(f"Vocabulary loaded from {path}. Size: {len(self)}")

    def encode(self, tokens):
        """
        Converts a list of tokens to indices.
        """
        unk_idx = self.stoi.get(Config.UNK_TOKEN, 1)
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def decode(self, indices):
        """
        Converts a list of indices to tokens.
        """
        return [
            self.itos[idx] if 0 <= idx < len(self.itos) else Config.UNK_TOKEN
            for idx in indices
        ]


def get_or_build_vocab(load_cached_data=True):
    """
    Retrieves the vocabulary.
    If load_cached_data is True and the file exists, loads it.
    Otherwise, builds it from the training corpus and saves it.
    """
    vocab_path = Config.VOCAB_PATH

    # 1. Try to load cached
    if load_cached_data and os.path.exists(vocab_path):
        print(f"Loading vocabulary from cache: {vocab_path}")
        vocab = Vocabulary()
        vocab.load(vocab_path)
        return vocab

    # 2. Build from scratch
    print("Building vocabulary from scratch...")

    # Load training data
    # We use the metadata file to get sentences
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA}"
        )

    # Read CSV
    # If DEBUG is on, we might want to limit rows, but for a robust vocab we usually want the full set.
    # However, to respect the time constraints and DEBUG flag if set:
    nrows = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None

    print(f"Reading training data from {Config.TRAIN_METADATA} (nrows={nrows})...")
    df = pd.read_csv(Config.TRAIN_METADATA, nrows=nrows)

    # Drop NaNs if any
    sentences = df["sentence"].dropna().tolist()

    vocab = Vocabulary()
    vocab.build(sentences)

    # 3. Save to cache
    vocab.save(vocab_path)

    return vocab
