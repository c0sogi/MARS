import os
import re
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class SimpleTokenizer:
    """
    A simple regex-based tokenizer that builds a vocabulary from training data.
    Cite Lesson 00005: Tokenization Hygiene.
    """

    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.pad_token_id = 0
        self.unk_token_id = 1

    def fit(self, texts):
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(str(text))
            counter.update(tokens)

        # Keep top most frequent words
        most_common = counter.most_common(self.vocab_size - 2)
        for token, _ in most_common:
            self.vocab[token] = len(self.vocab)
        print(f"Vocabulary built with {len(self.vocab)} tokens.")

    def tokenize(self, text):
        # Cite Lesson 00005: Regex-based alphanumeric extraction
        return re.findall(r"\b\w+\b", text.lower())

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text, max_length=None):
        tokens = self.tokenize(str(text))
        ids = self.convert_tokens_to_ids(tokens)
        if max_length:
            ids = ids[:max_length]
        return ids, len(tokens)


class ChatbotDataset(Dataset):
    """
    Dataset class for the Chatbot Arena task using Siamese LSTM.
    """

    def __init__(self, data, tokenizer, max_length=None, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else Config.MAX_LENGTH
        self.is_test = is_test

        self.ids = data["id"]
        self.prompts = data["prompt"]
        self.responses_a = data["response_a"]
        self.responses_b = data["response_b"]

        if not self.is_test:
            self.targets = data["targets"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        resp_a = str(self.responses_a[idx])
        resp_b = str(self.responses_b[idx])

        # Encode inputs
        # We encode them independently for the Siamese architecture
        ids_p, len_p_raw = self.tokenizer.encode(prompt, self.max_length)
        ids_a, len_a_raw = self.tokenizer.encode(resp_a, self.max_length)
        ids_b, len_b_raw = self.tokenizer.encode(resp_b, self.max_length)

        # Scalar features (Cite Lesson 00004)
        scalar_features = torch.tensor(
            [np.log1p(len_p_raw), np.log1p(len_a_raw), np.log1p(len_b_raw)],
            dtype=torch.float32,
        )

        # Convert to tensors
        t_p = torch.tensor(ids_p, dtype=torch.long)
        t_a = torch.tensor(ids_a, dtype=torch.long)
        t_b = torch.tensor(ids_b, dtype=torch.long)

        item = {
            "input_ids_p": t_p,
            "input_ids_a": t_a,
            "input_ids_b": t_b,
            "scalar_features": scalar_features,
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


class CollateFn:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Pad each sequence type separately
        ids_p = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids_p"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        ids_a = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids_a"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        ids_b = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids_b"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        scalar_features = torch.stack([item["scalar_features"] for item in batch])
        ids = [item["id"] for item in batch]

        batch_output = {
            "input_ids_p": ids_p,
            "input_ids_a": ids_a,
            "input_ids_b": ids_b,
            "scalar_features": scalar_features,
            "id": torch.tensor(ids, dtype=torch.long),
        }

        if "labels" in batch[0]:
            batch_output["labels"] = torch.stack([item["labels"] for item in batch])

        return batch_output


def load_and_cache_data(split, load_cached_data=True):
    """
    Loads data from CSV files or a cached NPZ file.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays of data.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            if "id" not in loaded.files:
                raise ValueError("Cache missing required 'id' key")
            data = {key: loaded[key] for key in loaded.files}
            return data
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Reloading from CSV.")

    # 2. Load from CSV
    if split == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        file_path = Config.VAL_DATA_PATH
    else:
        file_path = Config.TEST_DATA_PATH

    df = pd.read_csv(file_path)

    # Debugging subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SIZE)

    # Extract text columns and handle NaNs
    data = {
        "id": df["id"].values,
        "prompt": df["prompt"].fillna("").values.astype(str),
        "response_a": df["response_a"].fillna("").values.astype(str),
        "response_b": df["response_b"].fillna("").values.astype(str),
    }

    # Extract targets for train/val
    if split != "test":
        targets = df[Config.TARGET_COLS].values.astype(np.float32)
        data["targets"] = targets

    # 3. Save to cache
    np.savez(cache_path, **data)

    return data


def get_dataloaders(load_cached_data=True):
    """
    Initializes the Tokenizer, Datasets, and DataLoaders.
    """
    seed_everything(Config.SEED)

    # Load Data
    train_data = load_and_cache_data("train", load_cached_data)
    val_data = load_and_cache_data("val", load_cached_data)
    test_data = load_and_cache_data("test", load_cached_data)

    # Initialize and Fit Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=Config.VOCAB_SIZE)

    # Combine all text to fit tokenizer
    all_text = np.concatenate(
        [train_data["prompt"], train_data["response_a"], train_data["response_b"]]
    )
    tokenizer.fit(all_text)

    # Create Datasets
    train_dataset = ChatbotDataset(train_data, tokenizer, is_test=False)
    val_dataset = ChatbotDataset(val_data, tokenizer, is_test=False)
    test_dataset = ChatbotDataset(test_data, tokenizer, is_test=True)

    # Initialize Collate Function
    collate_fn = CollateFn(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
