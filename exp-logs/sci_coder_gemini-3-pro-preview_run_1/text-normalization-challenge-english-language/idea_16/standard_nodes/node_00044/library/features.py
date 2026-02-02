import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config


class FeatureExtractor:
    """
    Implements the Morphologically-Augmented feature extraction logic.
    Handles Character CNN inputs and Regex-based explicit features.
    """

    def __init__(self):
        self.regex_patterns = [re.compile(p) for p in Config.REGEX_PATTERNS]
        self.max_char_len = Config.MAX_CHAR_LEN
        self.char_vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.vocab_path = os.path.join(Config.VOCAB_DIR, "vocab_chars.json")

    def fit(self, tokens):
        """
        Builds character vocabulary from a list of tokens.
        """
        print("Fitting character vocabulary...")
        unique_chars = set()
        for token in tqdm(tokens, desc="Building Vocab"):
            unique_chars.update(str(token))

        # Sort for determinism
        for char in sorted(list(unique_chars)):
            if char not in self.char_vocab:
                self.char_vocab[char] = len(self.char_vocab)

        print(f"Vocabulary size: {len(self.char_vocab)}")
        self.save_vocab()

    def save_vocab(self):
        """Saves character vocabulary to JSON."""
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char_vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary to {self.vocab_path}")

    def load_vocab(self):
        """Loads character vocabulary from JSON."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(
                f"Vocabulary file not found at {self.vocab_path}. Call fit() on training data first."
            )

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.char_vocab = json.load(f)
        print(f"Loaded vocabulary of size {len(self.char_vocab)}")

    def transform(self, tokens):
        """
        Converts tokens into:
        1. Character ID sequences (padded/truncated)
        2. Regex binary feature vectors
        """
        num_tokens = len(tokens)

        # Initialize arrays
        char_features = np.zeros((num_tokens, self.max_char_len), dtype=np.int32)
        regex_features = np.zeros(
            (num_tokens, len(self.regex_patterns)), dtype=np.float32
        )

        print("Extracting features...")
        for i, token in enumerate(tqdm(tokens, desc="Transforming")):
            token_str = str(token)

            # 1. Character Features
            # Truncate if necessary
            chars = list(token_str)[: self.max_char_len]
            for j, char in enumerate(chars):
                char_features[i, j] = self.char_vocab.get(
                    char, self.char_vocab["<UNK>"]
                )

            # 2. Regex Features
            for k, pattern in enumerate(self.regex_patterns):
                if pattern.search(token_str):
                    regex_features[i, k] = 1.0

        return char_features, regex_features

    def get_vocab_size(self):
        return len(self.char_vocab)


def process_dataset(split_name, df=None, load_cached_data=True):
    """
    Orchestrates the feature extraction process with caching.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        df (pd.DataFrame, optional): Dataframe containing 'before' column.
                                     If None, loads from metadata.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing 'char_features', 'regex_features', 'tokens'
    """
    # Define cache paths
    cache_char_path = os.path.join(Config.CACHE_DIR, f"{split_name}_char_features.npy")
    cache_regex_path = os.path.join(
        Config.CACHE_DIR, f"{split_name}_regex_features.npy"
    )

    # Check cache
    if load_cached_data:
        if os.path.exists(cache_char_path) and os.path.exists(cache_regex_path):
            print(f"Loading cached features for {split_name}...")
            char_features = np.load(cache_char_path)
            regex_features = np.load(cache_regex_path)

            # We also need the raw tokens to return consistent output,
            # though usually the model only needs the arrays.
            # Loading the CSV just to get tokens if needed is fast compared to feature extraction.
            if df is None:
                csv_path = os.path.join(Config.METADATA_DIR, f"{split_name}.csv")
                df = pd.read_csv(csv_path, keep_default_na=False)

            # Ensure 'before' column is string
            tokens = df["before"].astype(str).tolist()

            return {
                "char_features": char_features,
                "regex_features": regex_features,
                "tokens": tokens,
            }
        else:
            print(f"Cache miss for {split_name}. Computing features...")

    # Load data if not provided
    if df is None:
        csv_path = os.path.join(Config.METADATA_DIR, f"{split_name}.csv")
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path, keep_default_na=False)

    tokens = df["before"].astype(str).tolist()

    # Initialize Extractor
    extractor = FeatureExtractor()

    # Handle Vocabulary
    if split_name == "train":
        # For training, we build the vocab
        extractor.fit(tokens)
    else:
        # For val/test, we must load the existing vocab
        try:
            extractor.load_vocab()
        except FileNotFoundError:
            # If we are running a test/val script but haven't trained yet,
            # we can't proceed properly. However, if this is a standalone run
            # where we might want to just process test data (e.g. submission),
            # we assume vocab exists. If not, we might need to fit on available data
            # but that's dangerous. We'll stick to strict logic: Train must run first.
            print(
                "Warning: Train vocab not found. If this is not 'train' split, this will fail."
            )
            raise

    # Transform
    char_features, regex_features = extractor.transform(tokens)

    # Save to Cache
    print(f"Saving features to {Config.CACHE_DIR}...")
    np.save(cache_char_path, char_features)
    np.save(cache_regex_path, regex_features)

    return {
        "char_features": char_features,
        "regex_features": regex_features,
        "tokens": tokens,
    }
