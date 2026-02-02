import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import library.config as config
from library.utils import set_seed

# =============================================================================
# CLASSES
# =============================================================================


class TextTokenizer:
    """
    Tokenizes text into integer sequences with a fixed vocabulary.
    Handles text cleaning, vocabulary building, and sequence padding.
    """

    def __init__(self, vocab_size=config.VOCAB_SIZE, max_len=config.MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        # Reserve 0 for padding, 1 for unknown words
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.frozen = False

    def clean_text(self, text):
        """Removes HTML tags and non-alphanumeric characters."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, texts):
        """Builds vocabulary from a list of texts."""
        if self.frozen:
            return

        counter = Counter()
        for text in texts:
            clean = self.clean_text(text)
            counter.update(clean.split())

        # -2 because we already have PAD and UNK
        most_common = counter.most_common(self.vocab_size - 2)

        for i, (word, _) in enumerate(most_common):
            idx = i + 2
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word

        self.frozen = True

    def transform(self, texts):
        """Converts texts to padded integer sequences."""
        token_ids_list = []
        for text in texts:
            clean = self.clean_text(text)
            words = clean.split()
            # Map words to indices, use UNK (1) if not found
            ids = [self.vocab.get(w, 1) for w in words]

            # Pad or Truncate
            if len(ids) > self.max_len:
                ids = ids[: self.max_len]
            else:
                ids = ids + [0] * (self.max_len - len(ids))

            token_ids_list.append(ids)

        return np.array(token_ids_list, dtype=np.int32)

    def save(self, path):
        """Saves vocabulary to JSON."""
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def load(self, path):
        """Loads vocabulary from JSON."""
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.frozen = True


class TagEncoder:
    """
    Encodes lists of tags into binary multi-hot vectors.
    """

    def __init__(self, top_k=config.TOP_K_TAGS):
        self.top_k = top_k
        self.tag_to_idx = {}
        self.classes_ = []
        self.frozen = False

    def fit(self, tags_list):
        """Identifies top-K tags from the training data."""
        if self.frozen:
            return

        counter = Counter()
        for tags in tags_list:
            if isinstance(tags, str):
                counter.update(tags.split())

        most_common = counter.most_common(self.top_k)
        self.classes_ = [t for t, _ in most_common]
        self.tag_to_idx = {t: i for i, t in enumerate(self.classes_)}
        self.frozen = True

    def transform(self, tags_list):
        """Converts tag strings to multi-hot vectors."""
        batch_size = len(tags_list)
        num_classes = len(self.classes_)
        y = np.zeros((batch_size, num_classes), dtype=np.float32)

        for i, tags in enumerate(tags_list):
            if isinstance(tags, str):
                for tag in tags.split():
                    if tag in self.tag_to_idx:
                        y[i, self.tag_to_idx[tag]] = 1.0
        return y

    def save(self, path):
        """Saves class mapping to JSON (avoids pickle)."""
        # Ensure we don't rely on pickle-based joblib
        json_path = os.path.splitext(path)[0] + ".json"
        data = {"classes": self.classes_}
        with open(json_path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Loads class mapping from JSON."""
        json_path = os.path.splitext(path)[0] + ".json"
        with open(json_path, "r") as f:
            data = json.load(f)
        self.classes_ = data["classes"]
        self.tag_to_idx = {t: i for i, t in enumerate(self.classes_)}
        self.frozen = True


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange data.
    """

    def __init__(self, tokens, labels=None, ids=None):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        self.ids = ids  # Keep as numpy array or list, not tensor, unless needed

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # If labels exist, return (tokens, labels) (Training/Validation)
        if self.labels is not None:
            return self.tokens[idx], self.labels[idx]

        # If IDs exist but no labels, return (tokens, id) (Testing)
        if self.ids is not None:
            return self.tokens[idx], self.ids[idx]

        return self.tokens[idx]


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================


def prepare_data(load_cached_data=True):
    """
    Main data processing pipeline.
    Checks cache -> Loads or Computes -> Saves -> Returns arrays.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_tokens": os.path.join(config.WORKING_DIR, "train_tokens.npy"),
        "train_labels": os.path.join(config.WORKING_DIR, "train_labels.npy"),
        "val_tokens": os.path.join(config.WORKING_DIR, "val_tokens.npy"),
        "val_labels": os.path.join(config.WORKING_DIR, "val_labels.npy"),
        "test_tokens": os.path.join(config.WORKING_DIR, "test_tokens.npy"),
        "test_ids": os.path.join(config.WORKING_DIR, "test_ids.npy"),
    }

    tokenizer_path = config.TOKENIZER_PATH
    encoder_path = config.LABEL_ENCODER_PATH

    # Check if we should and can load from cache
    all_exist = (
        all(os.path.exists(p) for p in cache_files.values())
        and os.path.exists(tokenizer_path)
        and os.path.exists(os.path.splitext(encoder_path)[0] + ".json")
    )

    # If debugging, we force re-computation to respect the sample size,
    # unless we want to implement a separate debug cache.
    # For safety, we skip cache loading in debug mode.
    if config.DEBUG_SAMPLE_SIZE is not None:
        load_cached_data = False

    if load_cached_data and all_exist:
        print("Loading data from cache...")
        train_tokens = np.load(cache_files["train_tokens"])
        train_labels = np.load(cache_files["train_labels"])
        val_tokens = np.load(cache_files["val_tokens"])
        val_labels = np.load(cache_files["val_labels"])
        test_tokens = np.load(cache_files["test_tokens"])
        test_ids = np.load(cache_files["test_ids"])

        tokenizer = TextTokenizer(vocab_size=config.VOCAB_SIZE, max_len=config.MAX_LEN)
        tokenizer.load(tokenizer_path)

        encoder = TagEncoder(top_k=config.TOP_K_TAGS)
        encoder.load(encoder_path)

        return (
            (train_tokens, train_labels),
            (val_tokens, val_labels),
            (test_tokens, test_ids),
            tokenizer,
            encoder,
        )

    print("Computing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    val_meta = pd.read_csv(config.VAL_META_PATH)
    test_meta = pd.read_csv(config.TEST_META_PATH)

    # Debugging: Downsample if requested
    if config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG MODE: Downsampling to {config.DEBUG_SAMPLE_SIZE} samples.")
        train_meta = train_meta.head(config.DEBUG_SAMPLE_SIZE)
        val_meta = val_meta.head(config.DEBUG_SAMPLE_SIZE)
        test_meta = test_meta.head(config.DEBUG_SAMPLE_SIZE)

    # 2. Load Raw Data
    # We load only necessary columns to save memory
    raw_train_path = os.path.join(config.INPUT_DIR, "train.csv")
    raw_test_path = os.path.join(config.INPUT_DIR, "test.csv")

    # Load raw train (contains Title, Body)
    # Note: Raw train contains Tags, but we rely on metadata Tags for the split ground truth
    df_raw_train = pd.read_csv(raw_train_path, usecols=["Id", "Title", "Body"])
    df_raw_test = pd.read_csv(raw_test_path, usecols=["Id", "Title", "Body"])

    # 3. Merge Metadata with Raw Data
    # Use left merge to preserve metadata split and order
    train_df = pd.merge(train_meta, df_raw_train, on="Id", how="left")
    val_df = pd.merge(val_meta, df_raw_train, on="Id", how="left")
    test_df = pd.merge(test_meta, df_raw_test, on="Id", how="left")

    # Handle missing text
    for df in [train_df, val_df, test_df]:
        df["Title"] = df["Title"].fillna("")
        df["Body"] = df["Body"].fillna("")
        if "Tags" in df.columns:
            df["Tags"] = df["Tags"].fillna("")

    # Prepare Text (Title + Body)
    train_texts = (train_df["Title"] + " " + train_df["Body"]).tolist()
    val_texts = (val_df["Title"] + " " + val_df["Body"]).tolist()
    test_texts = (test_df["Title"] + " " + test_df["Body"]).tolist()

    # 4. Tokenization
    print("Fitting Tokenizer...")
    tokenizer = TextTokenizer(vocab_size=config.VOCAB_SIZE, max_len=config.MAX_LEN)
    tokenizer.fit(train_texts)

    print("Transforming Text...")
    train_tokens = tokenizer.transform(train_texts)
    val_tokens = tokenizer.transform(val_texts)
    test_tokens = tokenizer.transform(test_texts)

    # 5. Label Encoding
    print("Fitting Tag Encoder...")
    encoder = TagEncoder(top_k=config.TOP_K_TAGS)
    encoder.fit(train_df["Tags"].tolist())

    print("Transforming Tags...")
    train_labels = encoder.transform(train_df["Tags"].tolist())
    val_labels = encoder.transform(val_df["Tags"].tolist())

    test_ids = test_df["Id"].values

    # 6. Save to Cache (Only if not debugging, to avoid overwriting full cache with debug data)
    if config.DEBUG_SAMPLE_SIZE is None:
        print("Saving processed data to cache...")
        np.save(cache_files["train_tokens"], train_tokens)
        np.save(cache_files["train_labels"], train_labels)
        np.save(cache_files["val_tokens"], val_tokens)
        np.save(cache_files["val_labels"], val_labels)
        np.save(cache_files["test_tokens"], test_tokens)
        np.save(cache_files["test_ids"], test_ids)

        tokenizer.save(tokenizer_path)
        encoder.save(encoder_path)

    return (
        (train_tokens, train_labels),
        (val_tokens, val_labels),
        (test_tokens, test_ids),
        tokenizer,
        encoder,
    )


def get_dataloaders(load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test sets.
    """
    set_seed(config.SEED)

    # Load/Process Data
    (train_X, train_y), (val_X, val_y), (test_X, test_ids), tokenizer, encoder = (
        prepare_data(load_cached_data)
    )

    # Create Datasets
    train_dataset = StackExchangeDataset(train_X, train_y)
    val_dataset = StackExchangeDataset(val_X, val_y)
    test_dataset = StackExchangeDataset(test_X, ids=test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, tokenizer, encoder
