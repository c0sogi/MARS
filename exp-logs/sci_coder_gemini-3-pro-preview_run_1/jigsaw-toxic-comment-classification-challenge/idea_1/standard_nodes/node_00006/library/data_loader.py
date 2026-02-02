import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import load_dataset


class TextTokenizer:
    """
    Custom tokenizer to convert text to integer sequences with padding.
    """

    def __init__(self, max_features=100000, max_len=200):
        self.max_features = max_features
        self.max_len = max_len
        self.word_index = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

    def _tokenize(self, text):
        # Cite solution_lesson_node_00004: Preserve punctuation as features
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def fit(self, texts):
        """
        Builds vocabulary from a list of texts.
        """
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Keep top max_features - 2 (for PAD and UNK)
        most_common = counter.most_common(self.max_features - 2)

        self.word_index = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        for i, (word, _) in enumerate(most_common):
            self.word_index[word] = i + 2

    def transform(self, texts):
        """
        Converts texts to fixed-length integer sequences.
        """
        num_samples = len(texts)
        sequences = np.full((num_samples, self.max_len), self.pad_idx, dtype=np.int32)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            # Convert tokens to indices
            seq = [self.word_index.get(t, self.unk_idx) for t in tokens]

            # Truncate if longer than max_len
            if len(seq) > self.max_len:
                sequences[i, :] = seq[: self.max_len]
            else:
                # Place at the beginning (post-padding is handled by initialization)
                sequences[i, : len(seq)] = seq

        return sequences

    def save(self, path):
        """Saves the word index to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.word_index, f)

    def load(self, path):
        """Loads the word index from a JSON file."""
        with open(path, "r") as f:
            self.word_index = json.load(f)


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return indices as LongTensor
        x_item = torch.tensor(self.X[idx], dtype=torch.long)

        if self.y is not None:
            # Return labels as FloatTensor
            y_item = torch.tensor(self.y[idx], dtype=torch.float)
            return x_item, y_item

        return x_item


def process_data(load_cached_data=True):
    """
    Handles data loading, tokenization, and caching.

    Returns:
        X_train, y_train, X_val, y_val, X_test, word_index
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "word_index": os.path.join(Config.CACHE_DIR, "word_index.json"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", Config.CACHE_DIR)
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])

        with open(cache_files["word_index"], "r") as f:
            word_index = json.load(f)

        return X_train, y_train, X_val, y_val, X_test, word_index

    print("Processing data from scratch...")

    # Load raw datasets using metadata
    # Note: val metadata uses train.csv as source
    train_df = load_dataset(Config.TRAIN_METADATA, Config.RAW_TRAIN_PATH)
    val_df = load_dataset(Config.VAL_METADATA, Config.RAW_TRAIN_PATH)
    test_df = load_dataset(Config.TEST_METADATA, Config.RAW_TEST_PATH)

    # Initialize and fit tokenizer
    tokenizer = TextTokenizer(max_features=Config.MAX_FEATURES, max_len=Config.MAX_LEN)
    print("Fitting tokenizer on training data...")
    tokenizer.fit(train_df["comment_text"].tolist())

    # Transform text to sequences
    print("Transforming text to sequences...")
    X_train = tokenizer.transform(train_df["comment_text"].tolist())
    X_val = tokenizer.transform(val_df["comment_text"].tolist())
    X_test = tokenizer.transform(test_df["comment_text"].tolist())

    # Extract labels
    y_train = train_df[Config.LABEL_COLS].values.astype(np.float32)
    y_val = val_df[Config.LABEL_COLS].values.astype(np.float32)

    # Save to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    tokenizer.save(cache_files["word_index"])

    return X_train, y_train, X_val, y_val, X_test, tokenizer.word_index


def get_dataloaders(debug=False, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data.
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader, word_index
    """
    # Load processed data
    X_train, y_train, X_val, y_val, X_test, word_index = process_data(load_cached_data)

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples.")
        X_train = X_train[: Config.DEBUG_SAMPLE_SIZE]
        y_train = y_train[: Config.DEBUG_SAMPLE_SIZE]
        X_val = X_val[: Config.DEBUG_SAMPLE_SIZE]
        y_val = y_val[: Config.DEBUG_SAMPLE_SIZE]
        X_test = X_test[: Config.DEBUG_SAMPLE_SIZE]

    # Create Datasets
    train_dataset = ToxicityDataset(X_train, y_train)
    val_dataset = ToxicityDataset(X_val, y_val)
    test_dataset = ToxicityDataset(X_test)  # No labels for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, word_index
