import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class EssayDataset(Dataset):
    """
    PyTorch Dataset for the Essay Scoring task.
    Wraps pre-tokenized inputs and labels for the Semantic Branch (DeBERTa).
    """

    def __init__(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids (np.ndarray): Array of token IDs.
            attention_mask (np.ndarray): Array of attention masks.
            labels (np.ndarray, optional): Array of target scores.
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tensors for a single sample.
        """
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.labels is not None:
            # Regression target: float32
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_essay_dataset(partition: str, load_cached_data: bool = True) -> EssayDataset:
    """
    Loads, tokenizes, and caches the dataset for a specific partition.
    Handles 'train', 'val', and 'test' splits using paths from Config.

    Args:
        partition (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        EssayDataset: An instantiated dataset ready for DataLoader.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file path
    cache_path = os.path.join(Config.CACHE_DIR, f"tokens_{partition}.npz")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {partition} dataset from cache: {cache_path}")
        try:
            with np.load(cache_path) as data:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                # Check if labels exist in the archive
                if "labels" in data:
                    labels = data["labels"]
                else:
                    labels = None
            return EssayDataset(input_ids, attention_mask, labels)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {partition} dataset...")

    # Load Metadata
    if partition == "train":
        df = pd.read_csv(Config.TRAIN_PATH)
    elif partition == "val":
        df = pd.read_csv(Config.VAL_PATH)
    elif partition == "test":
        df = pd.read_csv(Config.TEST_PATH)
    else:
        raise ValueError(f"Invalid partition: {partition}")

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"Debug mode enabled. Subsampling {partition} data...")
        df = df.head(50)

    # Preprocess Text: Minimal whitespace normalization
    # Filling NaNs to avoid tokenizer crashes
    texts = (
        df["full_text"]
        .fillna("")
        .astype(str)
        .apply(lambda x: " ".join(x.split()))
        .tolist()
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Tokenize
    print(f"Tokenizing {len(texts)} texts with {Config.MODEL_NAME}...")
    encodings = tokenizer(
        texts,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_attention_mask=True,
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Process Labels if available
    labels = None
    if "score" in df.columns:
        labels = df["score"].values.astype(np.float32)

    # 3. Save to cache (using np.savez to avoid pickle)
    print(f"Saving {partition} dataset to cache: {cache_path}")
    save_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
    if labels is not None:
        save_dict["labels"] = labels

    np.savez(cache_path, **save_dict)

    return EssayDataset(input_ids, attention_mask, labels)
