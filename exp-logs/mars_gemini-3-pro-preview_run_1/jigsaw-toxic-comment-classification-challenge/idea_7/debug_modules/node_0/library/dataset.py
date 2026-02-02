import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxic Comment Classification.
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


def _load_or_process_data(cfg, split_name, meta_path, tokenizer, load_cached_data):
    """
    Loads data from cache or processes it from scratch (metadata + raw text -> tokens).
    """
    # Define cache filenames with debug suffix if applicable
    suffix = "_debug" if cfg.debug else ""
    cache_dir = cfg.working_dir

    input_ids_path = os.path.join(cache_dir, f"{split_name}{suffix}_input_ids.npy")
    attn_mask_path = os.path.join(cache_dir, f"{split_name}{suffix}_attention_mask.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}{suffix}_labels.npy")

    # Determine if this split expects labels
    has_labels = split_name in ["train", "val"]

    # Check if cache exists
    cache_exists = os.path.exists(input_ids_path) and os.path.exists(attn_mask_path)
    if has_labels:
        cache_exists = cache_exists and os.path.exists(labels_path)

    # 1. Try to load from cache
    if load_cached_data and cache_exists:
        print(f"Loading {split_name} data from cache ({cache_dir})...")
        input_ids = np.load(input_ids_path)
        attention_mask = np.load(attn_mask_path)
        labels = np.load(labels_path) if has_labels else None
        return input_ids, attention_mask, labels

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch...")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Handle Debug Mode (Sample data)
    if cfg.debug:
        meta_df = meta_df.sample(
            n=min(1000, len(meta_df)), random_state=cfg.seed
        ).reset_index(drop=True)
        print(f"Debug mode: Sampled {len(meta_df)} rows from {split_name}.")

    # Merge with Raw Data
    # Identify unique source files referenced in metadata
    source_files = meta_df["source_file"].unique()
    merged_dfs = []

    for src_file in source_files:
        raw_file_path = os.path.join(cfg.input_dir, src_file)
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_file_path}")

        # Load raw data (only ID and Text needed)
        raw_df = pd.read_csv(raw_file_path, usecols=["id", "comment_text"])

        # Filter metadata for this source
        subset_meta = meta_df[meta_df["source_file"] == src_file]

        # Merge to attach text to metadata labels
        merged = pd.merge(subset_meta, raw_df, on="id", how="left")
        merged_dfs.append(merged)

    df = pd.concat(merged_dfs, ignore_index=True)

    # Handle missing text values
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    # Tokenize
    print(f"Tokenizing {len(df)} samples for {split_name}...")
    encoded = tokenizer.batch_encode_plus(
        df["comment_text"].tolist(),
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = None
    if has_labels:
        labels = df[cfg.target_cols].values.astype(np.float32)

    # 3. Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(input_ids_path, input_ids)
    np.save(attn_mask_path, attention_mask)
    if has_labels:
        np.save(labels_path, labels)

    return input_ids, attention_mask, labels


def create_dataloaders(cfg, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Process Train Data
    train_ids, train_mask, train_labels = _load_or_process_data(
        cfg, "train", cfg.train_meta_path, tokenizer, load_cached_data
    )

    # Process Validation Data
    val_ids, val_mask, val_labels = _load_or_process_data(
        cfg, "val", cfg.val_meta_path, tokenizer, load_cached_data
    )

    # Process Test Data
    test_ids, test_mask, _ = _load_or_process_data(
        cfg, "test", cfg.test_meta_path, tokenizer, load_cached_data
    )

    # Create Datasets
    train_dataset = ToxicityDataset(train_ids, train_mask, train_labels)
    val_dataset = ToxicityDataset(val_ids, val_mask, val_labels)
    test_dataset = ToxicityDataset(test_ids, test_mask, labels=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
