import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from library.config import Config
from library.utils import clean_text, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class Tokenizer:
    """
    Custom Tokenizer to convert text to fixed-length integer sequences.
    Handles vocabulary building, padding, and unknown tokens.
    """

    def __init__(
        self,
        max_len=Config.MAX_LEN,
        vocab_size=Config.VOCAB_SIZE,
        min_freq=Config.MIN_FREQ,
    ):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_fitted = False

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.split())

        # Select most common words
        # We reserve 2 slots for PAD and UNK
        most_common = counter.most_common(self.vocab_size - 2)

        idx = 2
        for word, freq in most_common:
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.vocab_fitted = True
        print(f"Vocabulary built. Size: {len(self.word2idx)}")

    def texts_to_sequences(self, texts):
        """
        Converts a list of strings to a numpy array of integer sequences.
        """
        if not self.vocab_fitted:
            raise ValueError(
                "Tokenizer must be fitted before calling texts_to_sequences."
            )

        sequences = []
        unk_idx = self.word2idx["<UNK>"]

        for text in texts:
            if not isinstance(text, str):
                seq = []
            else:
                words = text.split()
                # Map words to indices, use UNK if not found
                seq = [self.word2idx.get(w, unk_idx) for w in words]

            # Truncate or Pad
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]
            else:
                seq = seq + [0] * (self.max_len - len(seq))

            sequences.append(seq)

        return np.array(sequences, dtype=np.int32)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_fitted = True


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange data.
    """

    def __init__(self, tokens, labels=None, ids=None):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        self.ids = ids

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        item = {"tokens": self.tokens[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item


def _load_and_clean_raw_data(metadata_path, raw_data_path, is_test=False):
    """
    Helper to load metadata, merge with raw data, and clean text.
    """
    print(f"Loading data from {metadata_path} and {raw_data_path}...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    # Load raw data (optimize memory by selecting columns)
    cols = ["Id", "Title", "Body"]
    if not is_test:
        # For training raw file, Tags might be there but we rely on metadata for split
        # However, we need to merge.
        pass

    df_raw = pd.read_csv(raw_data_path, usecols=cols)

    # Merge
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    # Clean text
    print("Cleaning text...")
    # Combine Title and Body
    df["text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")
    df["text"] = df["text"].apply(clean_text)

    if not is_test:
        df["Tags"] = df["Tags"].fillna("")

    return df


def _process_tags(df_train, top_k=Config.TOP_K_TAGS):
    """
    Identifies top K tags and fits MultiLabelBinarizer.
    """
    print(f"Processing tags to find top {top_k}...")

    # Split tags into lists
    all_tags_list = df_train["Tags"].str.split().tolist()

    # Flatten and count
    all_tags_flat = [
        tag for tags in all_tags_list if isinstance(tags, list) for tag in tags
    ]
    tag_counts = Counter(all_tags_flat)

    # Get top K tags
    top_tags = {tag for tag, _ in tag_counts.most_common(top_k)}

    # Filter tags for each sample
    filtered_tags = []
    for tags in all_tags_list:
        if isinstance(tags, list):
            filtered_tags.append([t for t in tags if t in top_tags])
        else:
            filtered_tags.append([])

    # Fit MLB
    mlb = MultiLabelBinarizer()
    mlb.fit(filtered_tags)

    return mlb, top_tags


def prepare_data(load_cached_data=True):
    """
    Orchestrates the data processing pipeline with caching.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if all cache files exist
    cache_files = [
        Config.TRAIN_TOKENS_PATH,
        Config.TRAIN_LABELS_PATH,
        Config.VAL_TOKENS_PATH,
        Config.VAL_LABELS_PATH,
        Config.TEST_TOKENS_PATH,
        Config.TEST_IDS_PATH,
        Config.VOCAB_PATH,
        Config.MLB_PATH,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading cached data from disk...")
        train_tokens = np.load(Config.TRAIN_TOKENS_PATH)
        train_labels = np.load(Config.TRAIN_LABELS_PATH)
        val_tokens = np.load(Config.VAL_TOKENS_PATH)
        val_labels = np.load(Config.VAL_LABELS_PATH)
        test_tokens = np.load(Config.TEST_TOKENS_PATH)
        test_ids = np.load(Config.TEST_IDS_PATH)

        # Load artifacts
        tokenizer = Tokenizer()
        tokenizer.load(Config.VOCAB_PATH)
        mlb = joblib.load(Config.MLB_PATH)

        return (
            (train_tokens, train_labels),
            (val_tokens, val_labels),
            (test_tokens, test_ids),
            tokenizer,
            mlb,
        )

    print("Cache missing or reload requested. Processing data from scratch...")

    # --- 1. Process Training Data ---
    df_train = _load_and_clean_raw_data(
        Config.TRAIN_META_PATH, os.path.join(Config.INPUT_DIR, Config.TRAIN_RAW_FILE)
    )

    # Fit Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train["text"])
    tokenizer.save(Config.VOCAB_PATH)

    # Fit MLB
    mlb, top_tags_set = _process_tags(df_train)
    joblib.dump(mlb, Config.MLB_PATH)

    # Transform Train
    print("Transforming training data...")
    train_tokens = tokenizer.texts_to_sequences(df_train["text"])

    # Filter train tags to match top_tags_set before transform
    train_tags_list = (
        df_train["Tags"]
        .str.split()
        .apply(
            lambda x: [t for t in x if t in top_tags_set] if isinstance(x, list) else []
        )
    )
    train_labels = mlb.transform(train_tags_list)

    # Save Train
    np.save(Config.TRAIN_TOKENS_PATH, train_tokens)
    np.save(Config.TRAIN_LABELS_PATH, train_labels)

    # Free memory
    del df_train, train_tags_list

    # --- 2. Process Validation Data ---
    df_val = _load_and_clean_raw_data(
        Config.VAL_META_PATH, os.path.join(Config.INPUT_DIR, Config.TRAIN_RAW_FILE)
    )

    print("Transforming validation data...")
    val_tokens = tokenizer.texts_to_sequences(df_val["text"])

    val_tags_list = (
        df_val["Tags"]
        .str.split()
        .apply(
            lambda x: [t for t in x if t in top_tags_set] if isinstance(x, list) else []
        )
    )
    val_labels = mlb.transform(val_tags_list)

    np.save(Config.VAL_TOKENS_PATH, val_tokens)
    np.save(Config.VAL_LABELS_PATH, val_labels)

    del df_val, val_tags_list

    # --- 3. Process Test Data ---
    df_test = _load_and_clean_raw_data(
        Config.TEST_META_PATH,
        os.path.join(Config.INPUT_DIR, Config.TEST_RAW_FILE),
        is_test=True,
    )

    print("Transforming test data...")
    test_tokens = tokenizer.texts_to_sequences(df_test["text"])
    test_ids = df_test["Id"].values

    np.save(Config.TEST_TOKENS_PATH, test_tokens)
    np.save(Config.TEST_IDS_PATH, test_ids)

    return (
        (train_tokens, train_labels),
        (val_tokens, val_labels),
        (test_tokens, test_ids),
        tokenizer,
        mlb,
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    (
        (train_tokens, train_labels),
        (val_tokens, val_labels),
        (test_tokens, test_ids),
        tokenizer,
        mlb,
    ) = prepare_data(load_cached_data)

    print("Creating Datasets...")
    train_dataset = StackExchangeDataset(train_tokens, train_labels)
    val_dataset = StackExchangeDataset(val_tokens, val_labels)
    test_dataset = StackExchangeDataset(test_tokens, ids=test_ids)

    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer, mlb
