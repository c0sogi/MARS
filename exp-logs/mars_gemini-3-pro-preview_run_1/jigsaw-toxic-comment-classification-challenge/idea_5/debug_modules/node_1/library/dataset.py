import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    Dataset class for Toxicity Classification.
    Wraps pre-tokenized numpy arrays and serves them as PyTorch tensors.
    """

    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def load_and_preprocess(split_name, meta_path, load_cached_data=True):
    """
    Loads metadata, merges with raw text, tokenizes, and caches the result.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        meta_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (input_ids, attention_mask, labels)
    """
    # Determine cache suffix based on debug mode
    suffix = "_debug" if Config.debug else ""
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{split_name}{suffix}_input_ids.npy")
    mask_path = os.path.join(cache_dir, f"{split_name}{suffix}_attention_mask.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}{suffix}_labels.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(ids_path) and os.path.exists(mask_path):
        # Check labels existence (not needed for test, but good to check consistency)
        if split_name == "test" or os.path.exists(labels_path):
            print(f"Loading cached data for {split_name}{suffix} from {cache_dir}...")
            input_ids = np.load(ids_path)
            attention_mask = np.load(mask_path)
            labels = np.load(labels_path) if os.path.exists(labels_path) else None
            return input_ids, attention_mask, labels

    # 2. Process from Scratch
    print(f"Processing data for {split_name}{suffix}...")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    meta_df = pd.read_csv(meta_path)

    # Handle Debug Mode (Slice Metadata)
    if Config.debug:
        meta_df = meta_df.sample(
            n=min(1000, len(meta_df)), random_state=Config.seed
        ).reset_index(drop=True)
        print(f"Debug mode: sampled {len(meta_df)} rows for {split_name}.")

    # Identify Raw Source File
    # The metadata contains 'source_file', but we map it to Config paths for safety
    # Assuming train/val come from train.csv and test from test.csv
    if "train" in split_name or "val" in split_name:
        raw_source_path = Config.train_data_path
    else:
        raw_source_path = Config.test_data_path

    if not os.path.exists(raw_source_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_source_path}")

    raw_df = pd.read_csv(raw_source_path)

    # Merge Metadata with Raw Text
    # We select only necessary columns from raw_df to save memory during merge
    df = pd.merge(meta_df, raw_df[["id", "comment_text"]], on="id", how="left")

    # Fill Missing Text
    df["comment_text"] = df["comment_text"].fillna("none")

    # Tokenization
    print(f"Tokenizing {len(df)} texts for {split_name}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Batch Encode
    encoded = tokenizer.batch_encode_plus(
        df["comment_text"].tolist(),
        add_special_tokens=True,
        max_length=Config.max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Extract Labels
    labels = None
    if split_name != "test":
        labels = df[Config.target_cols].values.astype(np.float32)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(ids_path, input_ids)
    np.save(mask_path, attention_mask)
    if labels is not None:
        np.save(labels_path, labels)

    return input_ids, attention_mask, labels


def prepare_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Train Data
    train_ids, train_mask, train_labels = load_and_preprocess(
        "train", Config.train_meta_path, load_cached_data=load_cached_data
    )
    train_dataset = ToxicityDataset(train_ids, train_mask, train_labels)

    # Load Val Data
    val_ids, val_mask, val_labels = load_and_preprocess(
        "val", Config.val_meta_path, load_cached_data=load_cached_data
    )
    val_dataset = ToxicityDataset(val_ids, val_mask, val_labels)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size:   {len(val_dataset)}")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        DataLoader: The test data loader.
    """
    test_ids, test_mask, _ = load_and_preprocess(
        "test", Config.test_meta_path, load_cached_data=load_cached_data
    )

    test_dataset = ToxicityDataset(test_ids, test_mask, None)
    print(f"Test Dataset Size:  {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    return test_loader
