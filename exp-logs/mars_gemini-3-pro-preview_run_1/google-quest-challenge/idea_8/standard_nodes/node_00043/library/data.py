import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import config


class StackExchangeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract data to avoid pandas overhead in __getitem__
        self.q_texts = df["question_text"].tolist()
        self.a_texts = df["answer"].tolist()

        # Targets
        if not self.is_test:
            self.labels = df[config.target_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenize Question (Title + Body)
        # We do NOT pad here. Padding happens in the collate_fn for dynamic padding.
        q_enc = self.tokenizer(
            self.q_texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Tokenize Answer
        a_enc = self.tokenizer(
            self.a_texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract lists for padding
        q_features = [
            {"input_ids": b["q_input_ids"], "attention_mask": b["q_attention_mask"]}
            for b in batch
        ]
        a_features = [
            {"input_ids": b["a_input_ids"], "attention_mask": b["a_attention_mask"]}
            for b in batch
        ]

        # Dynamic padding using tokenizer
        q_batch = self.tokenizer.pad(q_features, padding=True, return_tensors="pt")
        a_batch = self.tokenizer.pad(a_features, padding=True, return_tensors="pt")

        batch_out = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
        }

        if "labels" in batch[0]:
            labels = torch.stack([b["labels"] for b in batch])
            batch_out["labels"] = labels

        return batch_out


def process_data(load_cached_data=True):
    """
    Loads raw data, processes text, and caches the result.
    """
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    meta_dims_cache = os.path.join(cache_dir, "meta_dims.npy")

    # 1. Try to load cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(meta_dims_cache)
        ):
            print("Loading cached processed data...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            meta_dims = np.load(meta_dims_cache, allow_pickle=True).item()
            return train_df, val_df, test_df, meta_dims

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load metadata splits
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    test_df = pd.read_csv(config.test_path)

    # Handle Text Columns (Fill NaNs)
    for df in [train_df, val_df, test_df]:
        df["question_title"] = df["question_title"].fillna("").astype(str)
        df["question_body"] = df["question_body"].fillna("").astype(str)
        df["answer"] = df["answer"].fillna("").astype(str)

        # Create combined Question text
        df["question_text"] = df["question_title"] + " " + df["question_body"]

    # No metadata encoding needed
    meta_dims = {}

    # Save to cache
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)
    np.save(meta_dims_cache, meta_dims)

    return train_df, val_df, test_df, meta_dims


def get_dataloaders(config, tokenizer, load_cached_data=True):
    """
    Factory function to get PyTorch DataLoaders.
    """
    # Load processed dataframes
    train_df, val_df, test_df, meta_dims = process_data(
        load_cached_data=load_cached_data
    )

    # Debug mode: sample subset
    if config.debug:
        print(f"DEBUG Mode: Sampling {config.debug_sample_size} rows.")
        train_df = train_df.iloc[: config.debug_sample_size]
        val_df = val_df.iloc[: config.debug_sample_size]
        # We generally keep test set intact or small sample for debug check
        test_df = test_df.iloc[: config.debug_sample_size]

    # Instantiate Datasets
    train_dataset = StackExchangeDataset(
        train_df, tokenizer, config.max_len, is_test=False
    )
    val_dataset = StackExchangeDataset(val_df, tokenizer, config.max_len, is_test=False)
    test_dataset = StackExchangeDataset(
        test_df, tokenizer, config.max_len, is_test=True
    )

    # Collate function for dynamic padding
    collate_fn = CollateFn(tokenizer)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, meta_dims
