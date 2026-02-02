import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything


def clean_text(text):
    """
    Cleans text by lowercasing and removing non-alphanumeric characters
    except for specific technical symbols (+, #, .).
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Replace anything that is NOT a lowercase letter, number, +, #, or . with a space
    # This preserves terms like 'c++', 'c#', '.net', 'node.js'
    text = re.sub(r"[^a-z0-9+#\.]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WordTokenizer:
    def __init__(self, vocab_size=50000, max_len=400):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_token_id = 0
        self.unk_token_id = 1

    def fit(self, texts):
        """
        Builds vocabulary from a list of texts.
        """
        print("Building vocabulary...")
        word_counts = Counter()
        for text in tqdm(texts, desc="Counting words"):
            word_counts.update(text.split())

        # Select top N words
        most_common = word_counts.most_common(
            self.vocab_size - 2
        )  # Reserve 2 for PAD, UNK

        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
        }
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx

        print(f"Vocabulary built with {len(self.word2idx)} tokens.")

    def transform(self, texts):
        """
        Converts texts to padded integer sequences.
        """
        print("Transforming texts to sequences...")
        num_samples = len(texts)
        # Pre-allocate array for efficiency
        sequences = np.full(
            (num_samples, self.max_len), self.pad_token_id, dtype=np.int32
        )

        for i, text in enumerate(tqdm(texts, desc="Tokenizing")):
            words = text.split()
            # Truncate if necessary
            if len(words) > self.max_len:
                words = words[: self.max_len]

            # Map to indices
            for j, word in enumerate(words):
                sequences[i, j] = self.word2idx.get(word, self.unk_token_id)

        return sequences

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                },
                f,
            )

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.vocab_size = data["vocab_size"]
            self.max_len = data["max_len"]


class StackExchangeDataset(Dataset):
    def __init__(self, tokens, labels=None):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tokens[idx], self.labels[idx]
        return self.tokens[idx]


def prepare_data(load_cached_data=True):
    """
    Loads data, processes it (cleaning, tokenization, encoding), and caches the result.
    Returns processed numpy arrays and objects.
    """
    Config.setup()

    # Check if cache exists
    cache_files = [
        Config.TRAIN_TOKENS_PATH,
        Config.TRAIN_LABELS_PATH,
        Config.VAL_TOKENS_PATH,
        Config.VAL_LABELS_PATH,
        Config.TEST_TOKENS_PATH,
        Config.MLB_PATH,
        Config.VOCAB_PATH,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_tokens = np.load(Config.TRAIN_TOKENS_PATH)
        train_labels = np.load(Config.TRAIN_LABELS_PATH)
        val_tokens = np.load(Config.VAL_TOKENS_PATH)
        val_labels = np.load(Config.VAL_LABELS_PATH)
        test_tokens = np.load(Config.TEST_TOKENS_PATH)
        mlb = joblib.load(Config.MLB_PATH)

        tokenizer = WordTokenizer()
        tokenizer.load(Config.VOCAB_PATH)

        return (
            train_tokens,
            train_labels,
            val_tokens,
            val_labels,
            test_tokens,
            mlb,
            tokenizer,
        )

    print("Cache missing or reload requested. Processing data from scratch...")

    # 1. Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_META_FILE)
    df_val_meta = pd.read_csv(Config.VAL_META_FILE)
    df_test_meta = pd.read_csv(Config.TEST_META_FILE)

    # 2. Load Raw Data
    # We load Id, Title, Body. Tags are in metadata for train/val.
    print("Loading raw text data...")
    # Using usecols to save memory
    raw_train = pd.read_csv(Config.TRAIN_RAW_FILE, usecols=["Id", "Title", "Body"])
    raw_test = pd.read_csv(Config.TEST_RAW_FILE, usecols=["Id", "Title", "Body"])

    # 3. Merge Metadata with Raw Data
    print("Merging metadata with raw text...")
    # Inner join ensures we only get rows defined in our split
    df_train = pd.merge(df_train_meta, raw_train, on="Id", how="inner")
    df_val = pd.merge(df_val_meta, raw_train, on="Id", how="inner")
    df_test = pd.merge(df_test_meta, raw_test, on="Id", how="inner")

    # Free up memory
    del raw_train, raw_test, df_train_meta, df_val_meta, df_test_meta
    import gc

    gc.collect()

    # 4. Text Preprocessing
    print("Preprocessing text...")

    def process_df_text(df):
        # Fill NA
        df["Title"] = df["Title"].fillna("")
        df["Body"] = df["Body"].fillna("")
        # Concatenate
        texts = (df["Title"] + " " + df["Body"]).astype(str)
        # Clean
        return [clean_text(t) for t in tqdm(texts, desc="Cleaning Text")]

    train_texts = process_df_text(df_train)
    val_texts = process_df_text(df_val)
    test_texts = process_df_text(df_test)

    # 5. Tokenization
    tokenizer = WordTokenizer(vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN)
    tokenizer.fit(train_texts)

    train_tokens = tokenizer.transform(train_texts)
    val_tokens = tokenizer.transform(val_texts)
    test_tokens = tokenizer.transform(test_texts)

    # Save Tokenizer
    tokenizer.save(Config.VOCAB_PATH)

    # 6. Label Processing
    print("Processing labels...")

    # Parse tags strings into lists
    df_train["tags_list"] = df_train["Tags"].fillna("").astype(str).str.split()
    df_val["tags_list"] = df_val["Tags"].fillna("").astype(str).str.split()

    # Identify Top K Tags
    all_tags = [tag for tags in df_train["tags_list"] for tag in tags]
    tag_counts = Counter(all_tags)
    top_tags = set([t for t, c in tag_counts.most_common(Config.NUM_TAGS)])

    print(f"Filtering top {Config.NUM_TAGS} tags...")

    def filter_tags(tag_list):
        return [t for t in tag_list if t in top_tags]

    train_tags_filtered = [
        filter_tags(tags)
        for tags in tqdm(df_train["tags_list"], desc="Filtering Train Tags")
    ]
    val_tags_filtered = [
        filter_tags(tags)
        for tags in tqdm(df_val["tags_list"], desc="Filtering Val Tags")
    ]

    # Binarize
    # Use sparse_output=True to avoid OOM during transform (Cite debug_lesson_5)
    mlb = MultiLabelBinarizer(classes=sorted(list(top_tags)), sparse_output=True)
    mlb.fit(
        train_tags_filtered
    )  # Should be same as classes provided, but good practice

    # Convert to dense int8 to save memory (20GB vs 80GB for float32)
    # StackExchangeDataset will cast to float32
    train_labels = mlb.transform(train_tags_filtered).astype(np.int8).toarray()
    val_labels = mlb.transform(val_tags_filtered).astype(np.int8).toarray()

    # Save Artifacts
    print("Saving processed data to cache...")
    np.save(Config.TRAIN_TOKENS_PATH, train_tokens)
    np.save(Config.TRAIN_LABELS_PATH, train_labels)
    np.save(Config.VAL_TOKENS_PATH, val_tokens)
    np.save(Config.VAL_LABELS_PATH, val_labels)
    np.save(Config.TEST_TOKENS_PATH, test_tokens)
    joblib.dump(mlb, Config.MLB_PATH)

    return (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        mlb,
        tokenizer,
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    seed_everything(Config.SEED)

    # Load/Process Data
    train_tokens, train_labels, val_tokens, val_labels, test_tokens, mlb, _ = (
        prepare_data(load_cached_data)
    )

    # Create Datasets
    train_dataset = StackExchangeDataset(train_tokens, train_labels)
    val_dataset = StackExchangeDataset(val_tokens, val_labels)
    test_dataset = StackExchangeDataset(test_tokens, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader, mlb
