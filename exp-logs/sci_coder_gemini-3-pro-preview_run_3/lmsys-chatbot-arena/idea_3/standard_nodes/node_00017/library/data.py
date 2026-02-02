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


class Vocabulary:
    def __init__(self, max_size=Config.VOCAB_SIZE):
        self.max_size = max_size
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = {0: "<pad>", 1: "<unk>"}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            # Cite solution_lesson_node_00005: "Tokenization Hygiene... regex-based alphanumeric extraction"
            tokens = re.findall(r"\b\w+\b", str(text).lower())
            counter.update(tokens)

        most_common = counter.most_common(self.max_size - 2)
        for i, (word, _) in enumerate(most_common):
            self.stoi[word] = i + 2
            self.itos[i + 2] = word

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path, "r") as f:
            self.stoi = json.load(f)
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, text, max_len):
        tokens = re.findall(r"\b\w+\b", str(text).lower())
        ids = [self.stoi.get(t, 1) for t in tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        return ids


class ChatbotDataset(Dataset):
    """
    Dataset class for the Chatbot Arena task using LSTM.
    """

    def __init__(self, data, vocab, max_length=None, is_test=False):
        self.data = data
        self.vocab = vocab
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

        # Encode text
        ids_prompt = self.vocab.encode(prompt, self.max_length)
        ids_a = self.vocab.encode(resp_a, self.max_length)
        ids_b = self.vocab.encode(resp_b, self.max_length)

        # Scalar features (Cite solution_lesson_node_00004: "Hybrid Inputs")
        # Cite solution_lesson_node_00005: "derive meta-features... from the processed token sequences"
        # We use the length of the non-padded tokens
        len_p = np.log1p(np.count_nonzero(ids_prompt))
        len_a = np.log1p(np.count_nonzero(ids_a))
        len_b = np.log1p(np.count_nonzero(ids_b))
        scalar_features = torch.tensor([len_p, len_a, len_b], dtype=torch.float32)

        item = {
            "ids_prompt": torch.tensor(ids_prompt, dtype=torch.long),
            "ids_a": torch.tensor(ids_a, dtype=torch.long),
            "ids_b": torch.tensor(ids_b, dtype=torch.long),
            "scalar_features": scalar_features,
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        ids_prompt = torch.stack([item["ids_prompt"] for item in batch])
        ids_a = torch.stack([item["ids_a"] for item in batch])
        ids_b = torch.stack([item["ids_b"] for item in batch])
        scalar_features = torch.stack([item["scalar_features"] for item in batch])
        ids = [item["id"] for item in batch]

        batch_output = {
            "ids_prompt": ids_prompt,
            "ids_a": ids_a,
            "ids_b": ids_b,
            "scalar_features": scalar_features,
            "id": torch.tensor(ids, dtype=torch.long),
        }

        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            batch_output["labels"] = labels

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

    Args:
        load_cached_data (bool): Whether to use cached data files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Load Data
    train_data = load_and_cache_data("train", load_cached_data)
    val_data = load_and_cache_data("val", load_cached_data)
    test_data = load_and_cache_data("test", load_cached_data)

    # Initialize and Build Vocabulary
    vocab_path = os.path.join(Config.WORKING_DIR, "vocab.json")
    vocab = Vocabulary(Config.VOCAB_SIZE)

    # We always rebuild vocab from train data to ensure consistency or load if exists
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        vocab.load(vocab_path)
    else:
        print("Building vocabulary from training data...")
        # Combine all text fields for vocab building
        all_text = np.concatenate(
            [train_data["prompt"], train_data["response_a"], train_data["response_b"]]
        )
        vocab.fit(all_text)
        vocab.save(vocab_path)
        print(f"Vocabulary built: {len(vocab)} tokens.")

    # Create Datasets
    train_dataset = ChatbotDataset(train_data, vocab, is_test=False)
    val_dataset = ChatbotDataset(val_data, vocab, is_test=False)
    test_dataset = ChatbotDataset(test_data, vocab, is_test=True)

    # Initialize Collate Function
    collate_fn = CollateFn()

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
