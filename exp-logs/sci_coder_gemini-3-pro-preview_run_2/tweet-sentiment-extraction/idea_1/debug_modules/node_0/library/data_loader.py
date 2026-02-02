import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    CACHE_DIR,
    MAX_LEN,
    SENTIMENT_MAP,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SIZE,
)
from library.utils import jaccard


class Vocabulary:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def build(self, texts):
        # Flatten all texts and count words
        words = []
        for text in texts:
            words.extend(str(text).split())

        # Sort by frequency for stability
        counter = Counter(words)
        unique_words = sorted(counter.keys())

        # 0 is reserved for padding, 1 for unknown
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        self.itos = {0: self.pad_token, 1: self.unk_token}

        idx = 2
        for w in unique_words:
            self.stoi[w] = idx
            self.itos[idx] = w
            idx += 1

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    def from_json(self, path):
        with open(path, "r") as f:
            self.stoi = json.load(f)
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, text):
        tokens = str(text).split()
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]


def find_best_span(text, selected_text):
    """
    Finds the start and end token indices in 'text' that best match 'selected_text'.
    First tries exact subsequence match, then falls back to maximizing Jaccard score.
    """
    # Tokenize
    text_tokens = str(text).split()

    # If selected_text is missing or empty
    if pd.isna(selected_text) or len(str(selected_text).strip()) == 0:
        return 0, 0

    # Optimization: If exact match exists as a sublist
    selected_tokens = str(selected_text).split()
    len_sel = len(selected_tokens)
    len_text = len(text_tokens)

    # Heuristic: Try to find exact sub-sequence first
    for i in range(len_text - len_sel + 1):
        if text_tokens[i : i + len_sel] == selected_tokens:
            return i, i + len_sel - 1

    # Fallback: Maximize Jaccard overlap
    best_jaccard = -1.0
    best_span = (0, 0)

    # We check all valid spans
    for i in range(len_text):
        for j in range(i, len_text):
            # Reconstruct substring from tokens
            candidate = " ".join(text_tokens[i : j + 1])
            score = jaccard(candidate, selected_text)
            if score > best_jaccard:
                best_jaccard = score
                best_span = (i, j)

    return best_span


def process_data(df, vocab, max_len, sentiment_map, is_test=False):
    """
    Converts dataframe text to padded indices and generates labels.
    """
    input_ids_list = []
    attention_masks_list = []
    sentiment_ids_list = []
    start_ids_list = []
    end_ids_list = []

    for _, row in df.iterrows():
        text = row["text"]
        sentiment = row["sentiment"]

        # Encode Text
        ids = vocab.encode(text)

        # Pad/Truncate
        curr_len = len(ids)
        if curr_len > max_len:
            ids = ids[:max_len]
            mask = [1] * max_len
        else:
            mask = [1] * curr_len + [0] * (max_len - curr_len)
            ids = ids + [vocab.stoi["<PAD>"]] * (max_len - curr_len)

        input_ids_list.append(ids)
        attention_masks_list.append(mask)
        sentiment_ids_list.append(sentiment_map[sentiment])

        if not is_test:
            selected_text = row["selected_text"]
            start, end = find_best_span(text, selected_text)

            # Clamp to max_len
            if start >= max_len:
                start = max_len - 1
            if end >= max_len:
                end = max_len - 1

            start_ids_list.append(start)
            end_ids_list.append(end)
        else:
            # Dummy labels for test set
            start_ids_list.append(0)
            end_ids_list.append(0)

    return {
        "input_ids": np.array(input_ids_list, dtype=np.int64),
        "attention_masks": np.array(attention_masks_list, dtype=np.int64),
        "sentiment_ids": np.array(sentiment_ids_list, dtype=np.int64),
        "start_ids": np.array(start_ids_list, dtype=np.int64),
        "end_ids": np.array(end_ids_list, dtype=np.int64),
    }


class TweetDataset(Dataset):
    def __init__(self, df, data_dict):
        self.text_ids = df["textID"].values
        self.texts = df["text"].values
        self.sentiments = df["sentiment"].values

        self.input_ids = torch.tensor(data_dict["input_ids"])
        self.attention_masks = torch.tensor(data_dict["attention_masks"])
        self.sentiment_ids = torch.tensor(data_dict["sentiment_ids"])
        self.start_ids = torch.tensor(data_dict["start_ids"])
        self.end_ids = torch.tensor(data_dict["end_ids"])

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "sentiment_id": self.sentiment_ids[idx],
            "start_idx": self.start_ids[idx],
            "end_idx": self.end_ids[idx],
            "text": str(self.texts[idx]),
            "textID": str(self.text_ids[idx]),
            "sentiment": str(self.sentiments[idx]),
        }


def get_loaders(load_cached_data=True):
    """
    Main entry point to get data loaders. Handles caching, vocabulary building, and batching.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load Dataframes from metadata
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Handle missing values (Metadata script should have handled this, but for safety)
    df_train.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)
    df_val.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)
    df_test.dropna(subset=["text", "sentiment"], inplace=True)

    # Debug Mode: Slice data if requested
    if DEBUG:
        df_train = df_train.head(DEBUG_SIZE)
        df_val = df_val.head(DEBUG_SIZE)
        df_test = df_test.head(DEBUG_SIZE)

    # Vocabulary Handling
    vocab_path = os.path.join(CACHE_DIR, "vocab.json")
    vocab = Vocabulary()

    # Load vocab if exists and caching is enabled
    if load_cached_data and os.path.exists(vocab_path):
        vocab.from_json(vocab_path)
    else:
        # Build vocab from training text and save
        vocab.build(df_train["text"].tolist())
        vocab.to_json(vocab_path)

    # Helper to process or load cache
    def get_cached_or_process(df, split_name, is_test=False):
        cache_file = os.path.join(CACHE_DIR, f"{split_name}_data.npz")

        if load_cached_data and os.path.exists(cache_file):
            try:
                loaded = np.load(cache_file)
                return {k: loaded[k] for k in loaded.files}
            except Exception:
                pass  # Fallback to processing if load fails

        # Process data
        data = process_data(df, vocab, MAX_LEN, SENTIMENT_MAP, is_test)

        # Save to cache
        np.savez(cache_file, **data)
        return data

    # Get processed data dictionaries
    train_data = get_cached_or_process(df_train, "train", is_test=False)
    val_data = get_cached_or_process(df_val, "val", is_test=False)
    test_data = get_cached_or_process(df_test, "test", is_test=True)

    # Create Datasets
    train_dataset = TweetDataset(df_train, train_data)
    val_dataset = TweetDataset(df_val, val_data)
    test_dataset = TweetDataset(df_test, test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab
