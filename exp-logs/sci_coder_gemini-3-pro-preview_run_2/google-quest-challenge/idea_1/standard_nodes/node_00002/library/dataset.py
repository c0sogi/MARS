import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config


# Ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class Vocabulary:
    """
    Handles mapping between words and integer indices.
    """

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build(self, texts, max_size):
        """
        Builds vocabulary from a list of text strings.
        """
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        # Reserve 0 for <PAD>, 1 for <UNK>
        # Select top (max_size - 2) frequent words
        most_common = word_counts.most_common(max_size - 2)

        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = {0: "<PAD>", 1: "<UNK>"}

        for idx, (word, _) in enumerate(most_common):
            self.stoi[word] = idx + 2
            self.itos[idx + 2] = word

        self.vocab_size = len(self.stoi)

    def save(self, path):
        """
        Saves vocabulary words to a numpy file.
        """
        # Create a list of words where index corresponds to the integer ID
        words = [self.itos[i] for i in range(self.vocab_size)]
        np.save(path, np.array(words))

    def load(self, path):
        """
        Loads vocabulary from a numpy file.
        """
        words = np.load(path)
        self.itos = {i: w for i, w in enumerate(words)}
        self.stoi = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def encode(self, text, max_len):
        """
        Converts a text string to a list of integer indices with padding/truncation.
        """
        tokens = text.split()
        indices = [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad
        if len(indices) < max_len:
            indices += [self.stoi["<PAD>"]] * (max_len - len(indices))

        return indices


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for the StackExchange task.
    """

    def __init__(self, q_data, a_data, y_data=None):
        self.q_data = torch.tensor(q_data, dtype=torch.long)
        self.a_data = torch.tensor(a_data, dtype=torch.long)
        if y_data is not None:
            self.y_data = torch.tensor(y_data, dtype=torch.float32)
        else:
            self.y_data = None

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):
        q = self.q_data[idx]
        a = self.a_data[idx]

        if self.y_data is not None:
            y = self.y_data[idx]
            return q, a, y
        else:
            # Return dummy targets for test set (all zeros)
            # Target dimension is 30
            dummy_y = torch.zeros(30, dtype=torch.float32)
            return q, a, dummy_y


def clean_text(text):
    """
    Basic text cleaning: lowercase.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def process_texts(df, vocab, max_len, fit_vocab=False, vocab_size=None):
    """
    Extracts text, cleans, (optionally) builds vocab, and encodes.
    """
    # Combine question title and body
    q_title = (
        df["question_title"].fillna("")
        if "question_title" in df.columns
        else pd.Series([""] * len(df))
    )
    q_body = (
        df["question_body"].fillna("")
        if "question_body" in df.columns
        else pd.Series([""] * len(df))
    )

    q_texts = (q_title + " " + q_body).apply(clean_text).tolist()
    a_texts = df["answer"].fillna("").apply(clean_text).tolist()

    if fit_vocab:
        all_texts = q_texts + a_texts
        vocab.build(all_texts, vocab_size)

    q_indices = [vocab.encode(t, max_len) for t in q_texts]
    a_indices = [vocab.encode(t, max_len) for t in a_texts]

    return np.array(q_indices), np.array(a_indices)


def prepare_data(load_cached_data=True, debug_sample_size=None):
    """
    Loads data, processes it, and handles caching.
    """
    Config.setup()  # Ensure directories exist

    cache_files = {
        "vocab": os.path.join(Config.CACHE_DIR, "vocab.npy"),
        "train_q": os.path.join(Config.CACHE_DIR, "train_q.npy"),
        "train_a": os.path.join(Config.CACHE_DIR, "train_a.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_q": os.path.join(Config.CACHE_DIR, "val_q.npy"),
        "val_a": os.path.join(Config.CACHE_DIR, "val_a.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_q": os.path.join(Config.CACHE_DIR, "test_q.npy"),
        "test_a": os.path.join(Config.CACHE_DIR, "test_a.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    all_cached = all(os.path.exists(p) for p in cache_files.values())

    vocab = Vocabulary()

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        vocab.load(cache_files["vocab"])
        train_q = np.load(cache_files["train_q"])
        train_a = np.load(cache_files["train_a"])
        train_y = np.load(cache_files["train_y"])
        val_q = np.load(cache_files["val_q"])
        val_a = np.load(cache_files["val_a"])
        val_y = np.load(cache_files["val_y"])
        test_q = np.load(cache_files["test_q"])
        test_a = np.load(cache_files["test_a"])
        test_ids = np.load(cache_files["test_ids"])
    else:
        print("Processing data from scratch...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Process Train (and build vocab)
        train_q, train_a = process_texts(
            train_df,
            vocab,
            Config.MAX_LEN,
            fit_vocab=True,
            vocab_size=Config.VOCAB_SIZE,
        )
        train_y = train_df[Config.TARGET_COLS].values.astype(np.float32)

        # Process Val
        val_q, val_a = process_texts(val_df, vocab, Config.MAX_LEN)
        val_y = val_df[Config.TARGET_COLS].values.astype(np.float32)

        # Process Test
        test_q, test_a = process_texts(test_df, vocab, Config.MAX_LEN)
        test_ids = test_df["qa_id"].values

        # Save to Cache
        print(f"Saving data to {Config.CACHE_DIR}...")
        vocab.save(cache_files["vocab"])
        np.save(cache_files["train_q"], train_q)
        np.save(cache_files["train_a"], train_a)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_q"], val_q)
        np.save(cache_files["val_a"], val_a)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_q"], test_q)
        np.save(cache_files["test_a"], test_a)
        np.save(cache_files["test_ids"], test_ids)

    # Handle Debugging
    if debug_sample_size is not None:
        train_q = train_q[:debug_sample_size]
        train_a = train_a[:debug_sample_size]
        train_y = train_y[:debug_sample_size]
        val_q = val_q[:debug_sample_size]
        val_a = val_a[:debug_sample_size]
        val_y = val_y[:debug_sample_size]
        test_q = test_q[:debug_sample_size]
        test_a = test_a[:debug_sample_size]
        test_ids = test_ids[:debug_sample_size]

    return {
        "train": (train_q, train_a, train_y),
        "val": (val_q, val_a, val_y),
        "test": (test_q, test_a, test_ids),
        "vocab": vocab,
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_sample_size=None
):
    """
    Returns DataLoaders for train, val, test, plus the vocab and test_ids.
    """
    data = prepare_data(load_cached_data, debug_sample_size)

    train_dataset = StackExchangeDataset(*data["train"])
    val_dataset = StackExchangeDataset(*data["val"])
    # Test dataset: pass None for y
    test_dataset = StackExchangeDataset(data["test"][0], data["test"][1], y_data=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, data["vocab"], data["test"][2]
