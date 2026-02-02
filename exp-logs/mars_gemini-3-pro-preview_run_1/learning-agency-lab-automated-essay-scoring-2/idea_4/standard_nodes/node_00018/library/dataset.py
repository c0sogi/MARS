import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    """

    def __init__(self, data, is_test=False):
        """
        Args:
            data (pd.DataFrame): DataFrame containing 'input_ids', 'attention_mask',
                                 'essay_id', and optionally 'score'.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.data = data
        self.is_test = is_test

        # Ensure input_ids and attention_mask are lists (in case of loading from parquet/numpy)
        self.input_ids = self.data["input_ids"].tolist()
        self.attention_mask = self.data["attention_mask"].tolist()
        self.essay_ids = self.data["essay_id"].tolist()

        if not self.is_test:
            self.labels = self.data["score"].astype(float).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "essay_id": self.essay_ids[idx],
        }

        if not self.is_test:
            # Return label as a float tensor for MSE Loss
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class CollateFn:
    """
    Custom collator to handle dynamic padding and pass through essay_ids.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract essay_ids and labels (if present) to handle separately
        essay_ids = [item.pop("essay_id") for item in batch]

        # Check if labels exist
        has_labels = "labels" in batch[0]
        labels = None
        if has_labels:
            labels = torch.stack([item.pop("labels") for item in batch])

        # Pad inputs using tokenizer
        # batch is now a list of dicts with only input_ids and attention_mask
        padded_batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")

        # Re-attach auxiliary data
        padded_batch["essay_id"] = essay_ids
        if has_labels:
            padded_batch["labels"] = labels

        return padded_batch


def preprocess_and_cache(
    csv_path, cache_path, tokenizer, cfg, is_test=False, load_cached_data=True
):
    """
    Loads data from CSV, tokenizes, and caches to Parquet.
    If cache exists and is requested, loads from cache.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            # Verify columns
            required_cols = ["essay_id", "input_ids", "attention_mask"]
            if not is_test:
                required_cols.append("score")
            if all(col in df.columns for col in required_cols):
                return df
            else:
                print("Cache corrupted or missing columns. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file {csv_path} not found.")

    df = pd.read_csv(csv_path)

    # Handle missing text
    df["full_text"] = df["full_text"].fillna("")

    # Tokenize
    # We use lists of texts for speed
    texts = df["full_text"].astype(str).tolist()

    # Tokenize with truncation but NO padding (save space in cache)
    # Padding will happen dynamically in the DataLoader
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=cfg.max_length,
        padding=False,
        return_attention_mask=True,
    )

    # Assign to DataFrame
    df["input_ids"] = encodings["input_ids"]
    df["attention_mask"] = encodings["attention_mask"]

    # Drop raw text to save space
    df = df.drop(columns=["full_text"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Save to parquet
    # PyArrow handles list columns efficiently
    print(f"Saving processed data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(cfg, load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        cfg: Configuration object.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(cfg.seed)

    # Initialize Tokenizer
    print(f"Initializing tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- Train Data ---
    train_df = preprocess_and_cache(
        csv_path=cfg.train_path,
        cache_path=cfg.train_cache_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # Debugging: Subset for fast debugging if enabled
    if cfg.debug:
        print("Debug mode: using subset of training data.")
        train_df = train_df.iloc[:100]

    train_dataset = EssayDataset(train_df, is_test=False)

    # --- Validation Data ---
    val_df = preprocess_and_cache(
        csv_path=cfg.val_path,
        cache_path=cfg.val_cache_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    if cfg.debug:
        val_df = val_df.iloc[:50]

    val_dataset = EssayDataset(val_df, is_test=False)

    # --- Test Data ---
    test_df = preprocess_and_cache(
        csv_path=cfg.test_path,
        cache_path=cfg.test_cache_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    if cfg.debug:
        test_df = test_df.iloc[:50]

    test_dataset = EssayDataset(test_df, is_test=True)

    # --- DataLoaders ---
    collate_fn = CollateFn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
