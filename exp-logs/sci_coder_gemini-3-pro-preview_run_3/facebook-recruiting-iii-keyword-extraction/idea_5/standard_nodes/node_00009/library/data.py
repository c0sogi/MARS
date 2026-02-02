import os
import re
import json
import numpy as np
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

from library.config import Config
from library.utils import get_logger

logger = get_logger("data_module")

# ==========================================
# 1. Text Processing Utilities
# ==========================================


def clean_text(text):
    """
    Cleans text by removing HTML tags and normalizing whitespace.
    Keeps alphanumeric characters and common programming symbols.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Replace non-alphanumeric (except +, #, -, .) with space
    # This preserves c++, c#, .net, node.js, etc.
    text = re.sub(r"[^a-z0-9+#.-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class Vocabulary:
    """
    Manages mapping between tokens and indices.
    """

    def __init__(self, vocab_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def fit(self, text_generator):
        """
        Builds vocabulary from an iterable of text strings.
        """
        logger.info("Building vocabulary...")
        counter = Counter()

        for text in text_generator:
            tokens = text.split()
            counter.update(tokens)

        # Filter by min_freq and select top vocab_size
        # Reserve 0 for PAD, 1 for UNK
        most_common = counter.most_common(self.vocab_size - 2)

        self.token2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2token = {0: self.pad_token, 1: self.unk_token}

        for idx, (token, count) in enumerate(most_common, start=2):
            if count < self.min_freq:
                break
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        logger.info(f"Vocabulary built. Size: {len(self.token2idx)}")

    def encode(self, text, max_len):
        """
        Converts text string to list of indices with padding/truncation.
        """
        tokens = text.split()
        indices = [
            self.token2idx.get(t, self.token2idx[self.unk_token]) for t in tokens
        ]

        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad
        if len(indices) < max_len:
            indices += [self.token2idx[self.pad_token]] * (max_len - len(indices))

        return indices

    def save(self, path):
        data = {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token,
            "params": {"vocab_size": self.vocab_size, "min_freq": self.min_freq},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.token2idx = data["token2idx"]
        # JSON keys are strings, convert back to int for idx2token
        self.idx2token = {int(k): v for k, v in data["idx2token"].items()}
        self.vocab_size = data["params"]["vocab_size"]
        self.min_freq = data["params"]["min_freq"]

    def __len__(self):
        return len(self.token2idx)


# ==========================================
# 2. Dataset Class
# ==========================================


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange Tag Prediction.
    Uses pre-computed numpy arrays for efficiency.
    """

    def __init__(self, title_ids, body_ids, labels=None, ids=None):
        self.title_ids = title_ids
        self.body_ids = body_ids
        self.labels = labels  # Scipy sparse matrix or None
        self.ids = ids  # Question Ids (for test set)

    def __len__(self):
        return len(self.title_ids)

    def __getitem__(self, idx):
        # Fetch features
        title = torch.tensor(self.title_ids[idx], dtype=torch.long)
        body = torch.tensor(self.body_ids[idx], dtype=torch.long)

        item = {"title": title, "body": body}

        if self.ids is not None:
            item["id"] = self.ids[idx]

        if self.labels is not None:
            # Convert sparse row to dense tensor on the fly
            # self.labels is a scipy.sparse matrix
            label_row = self.labels[idx].toarray().flatten()
            item["target"] = torch.tensor(label_row, dtype=torch.float32)

        return item


# ==========================================
# 3. Data Loading & Processing Pipeline
# ==========================================


def load_raw_data(metadata_path, raw_data_path):
    """
    Loads metadata and merges with raw text data.
    """
    logger.info(f"Loading metadata from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if Config.DEBUG:
        logger.info(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_meta = df_meta.sample(
            n=min(len(df_meta), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    logger.info(f"Loading raw data from {raw_data_path}...")
    # Read only necessary columns to save memory
    cols = ["Id", "Title", "Body"]
    try:
        df_raw = pd.read_csv(raw_data_path, usecols=cols)
    except ValueError:
        # Fallback if columns missing or different
        df_raw = pd.read_csv(raw_data_path)

    # Merge on Id
    logger.info("Merging metadata and raw data...")
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    # Fill NAs
    df["Title"] = df["Title"].fillna("")
    df["Body"] = df["Body"].fillna("")
    if "Tags" in df.columns:
        df["Tags"] = df["Tags"].fillna("")

    return df


def process_text_features(df, vocab):
    """
    Cleans, tokenizes and encodes text features.
    Returns numpy arrays for title and body.
    """
    logger.info("Processing text features...")

    num_samples = len(df)
    title_ids = np.zeros((num_samples, Config.MAX_TITLE_LEN), dtype=np.int32)
    body_ids = np.zeros((num_samples, Config.MAX_BODY_LEN), dtype=np.int32)

    # Pre-clean to speed up loop
    titles = df["Title"].apply(clean_text).values
    bodies = df["Body"].apply(clean_text).values

    for i in range(num_samples):
        title_ids[i] = vocab.encode(titles[i], Config.MAX_TITLE_LEN)
        body_ids[i] = vocab.encode(bodies[i], Config.MAX_BODY_LEN)

    return title_ids, body_ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main function to prepare data and return DataLoaders.
    Implements caching mechanism to avoid re-processing.
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "vocab": os.path.join(cache_dir, "vocab.json"),
        "mlb_classes": os.path.join(cache_dir, "mlb_classes.json"),
        "train_title": os.path.join(cache_dir, "train_title.npy"),
        "train_body": os.path.join(cache_dir, "train_body.npy"),
        "train_labels": os.path.join(cache_dir, "train_labels.npz"),
        "val_title": os.path.join(cache_dir, "val_title.npy"),
        "val_body": os.path.join(cache_dir, "val_body.npy"),
        "val_labels": os.path.join(cache_dir, "val_labels.npz"),
        "test_title": os.path.join(cache_dir, "test_title.npy"),
        "test_body": os.path.join(cache_dir, "test_body.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        logger.info("Loading cached data from disk...")

        # Load Vocab
        vocab = Vocabulary()
        vocab.load(files["vocab"])

        # Load MLB
        with open(files["mlb_classes"], "r") as f:
            mlb_classes = json.load(f)
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit([mlb_classes])  # Hack to initialize classes_

        # Load Arrays
        train_title = np.load(files["train_title"])
        train_body = np.load(files["train_body"])
        train_labels = sparse.load_npz(files["train_labels"])

        val_title = np.load(files["val_title"])
        val_body = np.load(files["val_body"])
        val_labels = sparse.load_npz(files["val_labels"])

        test_title = np.load(files["test_title"])
        test_body = np.load(files["test_body"])
        test_ids = np.load(files["test_ids"])

    else:
        logger.info("Cache not found or ignored. Processing data from scratch...")

        # 1. Load DataFrames
        df_train = load_raw_data(Config.TRAIN_META_PATH, Config.TRAIN_RAW_PATH)
        df_val = load_raw_data(Config.VAL_META_PATH, Config.TRAIN_RAW_PATH)
        df_test = load_raw_data(Config.TEST_META_PATH, Config.TEST_RAW_PATH)

        # 2. Build Vocabulary
        # We use a generator to feed the fit function to optimize memory
        def text_generator(dfs):
            for df in dfs:
                for col in ["Title", "Body"]:
                    for text in df[col]:
                        yield clean_text(text)

        vocab = Vocabulary(vocab_size=Config.VOCAB_SIZE, min_freq=Config.MIN_FREQ)
        # Fit on training data only to avoid leakage, or train+val if preferred.
        # Using train is standard.
        vocab.fit(text_generator([df_train]))
        vocab.save(files["vocab"])

        # 3. Fit MLB
        logger.info("Fitting MultiLabelBinarizer...")
        train_tags = df_train["Tags"].str.split()
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit(train_tags)

        # Save MLB classes to JSON (avoiding pickle for data persistence)
        with open(files["mlb_classes"], "w") as f:
            json.dump(mlb.classes_.tolist(), f)

        # 4. Transform and Save Data

        # Train
        train_title, train_body = process_text_features(df_train, vocab)
        train_labels = mlb.transform(train_tags)
        np.save(files["train_title"], train_title)
        np.save(files["train_body"], train_body)
        sparse.save_npz(files["train_labels"], train_labels)

        # Val
        val_title, val_body = process_text_features(df_val, vocab)
        val_tags = df_val["Tags"].str.split()
        val_labels = mlb.transform(val_tags)
        np.save(files["val_title"], val_title)
        np.save(files["val_body"], val_body)
        sparse.save_npz(files["val_labels"], val_labels)

        # Test
        test_title, test_body = process_text_features(df_test, vocab)
        test_ids = df_test["Id"].values
        np.save(files["test_title"], test_title)
        np.save(files["test_body"], test_body)
        np.save(files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = StackExchangeDataset(train_title, train_body, train_labels)
    val_dataset = StackExchangeDataset(val_title, val_body, val_labels)
    test_dataset = StackExchangeDataset(test_title, test_body, ids=test_ids)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, vocab, mlb
