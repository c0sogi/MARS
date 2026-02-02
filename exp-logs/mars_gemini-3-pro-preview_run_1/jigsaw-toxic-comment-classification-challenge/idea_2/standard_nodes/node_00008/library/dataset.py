import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification using RoBERTa.
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def load_and_preprocess_data(split, tokenizer, load_cached_data=True):
    """
    Loads data, tokenizes it, and implements caching logic using numpy files.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer: Pre-trained tokenizer instance.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (encodings dict, labels numpy array or None)
    """
    # Define cache paths
    # Append _debug to filename if in debug mode to avoid polluting full cache
    suffix = "_debug" if Config.DEBUG else ""
    cache_prefix = os.path.join(Config.WORKING_DIR, f"{split}{suffix}")

    input_ids_path = f"{cache_prefix}_input_ids.npy"
    mask_path = f"{cache_prefix}_attention_mask.npy"
    labels_path = f"{cache_prefix}_labels.npy"

    # Check if all required cache files exist
    # Labels are not required for 'test' split
    cache_exists = (
        os.path.exists(input_ids_path)
        and os.path.exists(mask_path)
        and (split == "test" or os.path.exists(labels_path))
    )

    # 1. Try to load from cache
    if load_cached_data and cache_exists:
        print(f"Loading cached data for '{split}' split from {Config.WORKING_DIR}...")
        input_ids = np.load(input_ids_path)
        attention_mask = np.load(mask_path)
        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}

        labels = None
        if split != "test":
            labels = np.load(labels_path)

        return encodings, labels

    # 2. Compute/Process from scratch
    print(f"Processing data for '{split}' split (Cache miss or force reload)...")

    # Determine Metadata and Raw paths
    if split == "train":
        meta_path = Config.TRAIN_METADATA
        raw_path = Config.TRAIN_RAW
    elif split == "val":
        meta_path = Config.VAL_METADATA
        raw_path = Config.TRAIN_RAW
    elif split == "test":
        meta_path = Config.TEST_METADATA
        raw_path = Config.TEST_RAW
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    meta_df = pd.read_csv(meta_path)

    # Load Raw Data (only necessary columns)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    raw_df = pd.read_csv(raw_path, usecols=["id", "comment_text"])

    # Merge Metadata with Raw Text
    # We use left join on 'id' to attach text to the specific split's metadata
    df = pd.merge(meta_df, raw_df, on="id", how="left")

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("")

    # Apply Debug Sampling if enabled
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling first {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Tokenize
    print("Tokenizing texts...")
    encodings = tokenizer(
        df["comment_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LEN,
        return_tensors="np",
    )

    # Save to Cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(input_ids_path, encodings["input_ids"])
    np.save(mask_path, encodings["attention_mask"])

    labels = None
    if split != "test":
        labels = df[Config.LABEL_COLS].values
        np.save(labels_path, labels)

    return encodings, labels


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for training and validation sets.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    print("Initializing Tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(Config.MODEL_NAME)

    # Prepare Train Data
    train_encodings, train_labels = load_and_preprocess_data(
        "train", tokenizer, load_cached_data
    )
    train_dataset = ToxicityDataset(train_encodings, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Prepare Validation Data
    val_encodings, val_labels = load_and_preprocess_data(
        "val", tokenizer, load_cached_data
    )
    val_dataset = ToxicityDataset(val_encodings, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, tokenizer


def get_test_dataloader(tokenizer, load_cached_data=True):
    """
    Creates and returns DataLoader for the test set.

    Args:
        tokenizer: The tokenizer instance used for training data.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: Test data loader.
    """
    test_encodings, _ = load_and_preprocess_data("test", tokenizer, load_cached_data)
    test_dataset = ToxicityDataset(test_encodings, labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    return test_loader
