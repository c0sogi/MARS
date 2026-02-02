import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, logging

# Import configuration and utilities from the provided library files
from library.config import Config
from library.utils import seed_everything

# Suppress transformer warnings for cleaner output
logging.set_verbosity_error()


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Wraps pre-tokenized numpy arrays and returns tensors.
    """

    def __init__(
        self, input_ids, attention_masks, targets=None, identities=None, sample_ids=None
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.identities = identities
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "id": self.sample_ids[idx],
        }

        if self.targets is not None:
            # Target is a float fraction [0, 1]
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.identities is not None:
            # Identities are a float vector
            item["identities"] = torch.tensor(self.identities[idx], dtype=torch.float)

        return item


def prepare_data(split, load_cached_data=True, tokenizer=None):
    """
    Loads, merges, tokenizes, and caches data for a specific split.

    Args:
        split (str): One of 'train', 'validation', 'test'.
        load_cached_data (bool): If True, attempts to load from .npy cache.
        tokenizer (AutoTokenizer): Pre-loaded tokenizer instance.

    Returns:
        tuple: (input_ids, attention_masks, targets, identities, sample_ids)
               targets and identities are None for 'test' split.
    """
    # Define cache file paths
    cache_prefix = os.path.join(Config.WORKING_DIR, f"{split}")
    path_ids = f"{cache_prefix}_ids.npy"
    path_input_ids = f"{cache_prefix}_input_ids.npy"
    path_masks = f"{cache_prefix}_masks.npy"
    path_targets = f"{cache_prefix}_targets.npy"
    path_identities = f"{cache_prefix}_identities.npy"

    # Check if cache exists
    files_exist = (
        os.path.exists(path_ids)
        and os.path.exists(path_input_ids)
        and os.path.exists(path_masks)
    )

    if split != "test":
        files_exist = (
            files_exist
            and os.path.exists(path_targets)
            and os.path.exists(path_identities)
        )

    # 1. Load from Cache
    if load_cached_data and files_exist:
        print(f"Loading cached {split} data from {Config.WORKING_DIR}...")
        sample_ids = np.load(path_ids)
        input_ids = np.load(path_input_ids)
        attention_masks = np.load(path_masks)

        targets = np.load(path_targets) if split != "test" else None
        identities = np.load(path_identities) if split != "test" else None

        return input_ids, attention_masks, targets, identities, sample_ids

    # 2. Process from Scratch
    print(f"Processing {split} data from scratch...")

    # Load Metadata
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "validation":
        meta_path = Config.VALID_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df_meta = pd.read_csv(meta_path)

    # Identify Source Text File
    # Metadata has 'source_file' column, but we know train/val come from train.csv and test from test.csv
    # To be efficient, we load the specific source file needed.
    source_file_name = df_meta["source_file"].iloc[0]
    source_path = os.path.join(Config.INPUT_DIR, source_file_name)

    # Load Text Data
    # We only read 'id' and 'comment_text' to save memory
    print(f"Loading raw text from {source_path}...")
    df_text = pd.read_csv(source_path, usecols=["id", "comment_text"])

    # Merge Metadata with Text
    print("Merging metadata with text...")
    df = df_meta.merge(df_text, on="id", how="inner")

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("")

    # Tokenization
    print("Tokenizing...")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        # Save tokenizer for reproducibility
        tokenizer.save_pretrained(Config.TOKENIZER_CACHE_DIR)

    # We use batch_encode_plus for speed.
    # Pad to max_length to create fixed size arrays (Pad-then-Trim strategy preparation)
    encoded = tokenizer.batch_encode_plus(
        df["comment_text"].tolist(),
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    sample_ids = df["id"].values

    targets = None
    identities = None

    if split != "test":
        targets = df[Config.TARGET_COL].values.astype(np.float32)
        # Extract identity columns
        identities = df[Config.IDENTITY_COLUMNS].values.astype(np.float32)
        # Handle NaNs in identity columns (treat as 0.0 if missing)
        identities = np.nan_to_num(identities, nan=0.0)

    # Save to Cache
    print(f"Saving {split} data to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    np.save(path_ids, sample_ids)
    np.save(path_input_ids, input_ids)
    np.save(path_masks, attention_masks)

    if split != "test":
        np.save(path_targets, targets)
        np.save(path_identities, identities)

    return input_ids, attention_masks, targets, identities, sample_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Load Tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # --------------------------------------------------------------------------
    # Train Set
    # --------------------------------------------------------------------------
    train_inputs, train_masks, train_targets, train_identities, train_ids = (
        prepare_data("train", load_cached_data, tokenizer)
    )

    train_dataset = ToxicityDataset(
        train_inputs, train_masks, train_targets, train_identities, train_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------------------------
    # Validation Set
    # --------------------------------------------------------------------------
    val_inputs, val_masks, val_targets, val_identities, val_ids = prepare_data(
        "validation", load_cached_data, tokenizer
    )

    val_dataset = ToxicityDataset(
        val_inputs, val_masks, val_targets, val_identities, val_ids
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # --------------------------------------------------------------------------
    # Test Set
    # --------------------------------------------------------------------------
    test_inputs, test_masks, _, _, test_ids = prepare_data(
        "test", load_cached_data, tokenizer
    )

    test_dataset = ToxicityDataset(
        test_inputs, test_masks, targets=None, identities=None, sample_ids=test_ids
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
