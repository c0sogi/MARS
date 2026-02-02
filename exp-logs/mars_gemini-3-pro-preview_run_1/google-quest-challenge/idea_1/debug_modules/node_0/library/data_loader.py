import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config


class Tokenizer:
    def __init__(self, vocab_size=Config.VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

    def fit(self, texts):
        """
        Builds vocabulary from a list of text strings.
        """
        counter = Counter()
        for text in texts:
            if not isinstance(text, str):
                continue
            if Config.TOKENIZER_LOWER:
                text = text.lower()
            tokens = text.split()
            counter.update(tokens)

        # Start with special tokens
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}

        # Add most common words
        # We reserve 2 slots for PAD and UNK
        most_common = counter.most_common(self.vocab_size - 2)

        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)

    def transform(self, texts, max_len):
        """
        Converts a list of texts to a numpy matrix of shape (len(texts), max_len).
        """
        seqs = []
        for text in texts:
            if not isinstance(text, str):
                text = ""
            if Config.TOKENIZER_LOWER:
                text = text.lower()
            tokens = text.split()

            # Convert to indices
            seq = [self.word2idx.get(token, self.unk_idx) for token in tokens]

            # Truncate or Pad
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [self.pad_idx] * (max_len - len(seq))

            seqs.append(seq)

        return np.array(seqs, dtype=np.int32)

    def save(self, path):
        """
        Saves the vocabulary (list of words) to a .npy file.
        The index is implicit in the order.
        """
        # Create a list where index i contains the word for ID i
        sorted_vocab = sorted(self.word2idx.items(), key=lambda item: item[1])
        words = [item[0] for item in sorted_vocab]
        np.save(path, np.array(words))

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        words = np.load(path)
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.pad_idx = self.word2idx.get(self.pad_token, 0)
        self.unk_idx = self.word2idx.get(self.unk_token, 1)


class StackExchangeDataset(Dataset):
    def __init__(self, q_data, a_data, targets=None):
        self.q_data = torch.tensor(q_data, dtype=torch.long)
        self.a_data = torch.tensor(a_data, dtype=torch.long)
        self.targets = None
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.q_data[idx], self.a_data[idx], self.targets[idx]
        else:
            return self.q_data[idx], self.a_data[idx]


def _process_text_column(df, col_name):
    return df[col_name].fillna("").astype(str).tolist()


def _load_and_process_data():
    """
    Internal function to load raw CSVs, fit tokenizer, and process data.
    Returns dictionary of arrays and the tokenizer.
    """
    # Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Prepare Text
    # Question = Title + " " + Body
    train_q_text = (
        train_df["question_title"].fillna("")
        + " "
        + train_df["question_body"].fillna("")
    ).tolist()
    val_q_text = (
        val_df["question_title"].fillna("") + " " + val_df["question_body"].fillna("")
    ).tolist()
    test_q_text = (
        test_df["question_title"].fillna("") + " " + test_df["question_body"].fillna("")
    ).tolist()

    train_a_text = train_df["answer"].fillna("").tolist()
    val_a_text = val_df["answer"].fillna("").tolist()
    test_a_text = test_df["answer"].fillna("").tolist()

    # Fit Tokenizer on Train Data (Question + Answer)
    tokenizer = Tokenizer(vocab_size=Config.VOCAB_SIZE)
    tokenizer.fit(train_q_text + train_a_text)

    # Transform
    train_q = tokenizer.transform(train_q_text, Config.MAX_LEN_Q)
    val_q = tokenizer.transform(val_q_text, Config.MAX_LEN_Q)
    test_q = tokenizer.transform(test_q_text, Config.MAX_LEN_Q)

    train_a = tokenizer.transform(train_a_text, Config.MAX_LEN_A)
    val_a = tokenizer.transform(val_a_text, Config.MAX_LEN_A)
    test_a = tokenizer.transform(test_a_text, Config.MAX_LEN_A)

    # Targets
    target_cols = Config.TARGET_COLS
    train_y = train_df[target_cols].values.astype(np.float32)
    val_y = val_df[target_cols].values.astype(np.float32)

    return {
        "train_q": train_q,
        "train_a": train_a,
        "train_y": train_y,
        "val_q": val_q,
        "val_a": val_a,
        "val_y": val_y,
        "test_q": test_q,
        "test_a": test_a,
    }, tokenizer


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
):
    """
    Main function to get DataLoaders. Handles caching and debug slicing.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "vocab": os.path.join(cache_dir, "vocab.npy"),
        "train_q": os.path.join(cache_dir, "train_q.npy"),
        "train_a": os.path.join(cache_dir, "train_a.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_q": os.path.join(cache_dir, "val_q.npy"),
        "val_a": os.path.join(cache_dir, "val_a.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_q": os.path.join(cache_dir, "test_q.npy"),
        "test_a": os.path.join(cache_dir, "test_a.npy"),
    }

    data = {}
    tokenizer = Tokenizer(vocab_size=Config.VOCAB_SIZE)

    # Check if cache exists
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        try:
            # Load data
            data["train_q"] = np.load(files["train_q"])
            data["train_a"] = np.load(files["train_a"])
            data["train_y"] = np.load(files["train_y"])
            data["val_q"] = np.load(files["val_q"])
            data["val_a"] = np.load(files["val_a"])
            data["val_y"] = np.load(files["val_y"])
            data["test_q"] = np.load(files["test_q"])
            data["test_a"] = np.load(files["test_a"])

            # Load tokenizer
            tokenizer.load(files["vocab"])

        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")
            load_cached_data = False  # Fallback to processing

    if not load_cached_data or not cache_exists:
        data, tokenizer = _load_and_process_data()

        # Save to cache
        np.save(files["train_q"], data["train_q"])
        np.save(files["train_a"], data["train_a"])
        np.save(files["train_y"], data["train_y"])
        np.save(files["val_q"], data["val_q"])
        np.save(files["val_a"], data["val_a"])
        np.save(files["val_y"], data["val_y"])
        np.save(files["test_q"], data["test_q"])
        np.save(files["test_a"], data["test_a"])
        tokenizer.save(files["vocab"])

    # Debug Slicing
    if debug:
        limit = Config.DEBUG_SIZE
        data["train_q"] = data["train_q"][:limit]
        data["train_a"] = data["train_a"][:limit]
        data["train_y"] = data["train_y"][:limit]
        data["val_q"] = data["val_q"][:limit]
        data["val_a"] = data["val_a"][:limit]
        data["val_y"] = data["val_y"][:limit]
        data["test_q"] = data["test_q"][:limit]
        data["test_a"] = data["test_a"][:limit]

    # Create Datasets
    train_dataset = StackExchangeDataset(
        data["train_q"], data["train_a"], data["train_y"]
    )
    val_dataset = StackExchangeDataset(data["val_q"], data["val_a"], data["val_y"])
    test_dataset = StackExchangeDataset(data["test_q"], data["test_a"], targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
