import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from library.config import Config
from library.cpc_utils import get_cpc_texts


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for Phrase Matching.
    Constructs input: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to lists for efficient access
        self.anchors = data["anchor"].values
        self.targets = data["target"].values
        self.contexts = data["context_text"].values
        self.ids = data["id"].values

        if not is_test:
            self.scores = data["score"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct the first segment: Context + [SEP] + Anchor
        # We manually insert the separator token between Context and Anchor.
        # The tokenizer will automatically add [CLS] at start, [SEP] between segments, and [SEP] at end.
        # Resulting structure: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        sep = self.tokenizer.sep_token
        first_segment = f"{context} {sep} {anchor}"
        second_segment = target

        inputs = self.tokenizer(
            first_segment,
            second_segment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


def process_data(cfg: Config, load_cached_data: bool = True):
    """
    Loads training and validation metadata, merges them into a single training set,
    adds CPC context text, and creates stratified folds.

    Implements strict caching logic using Parquet.
    """
    cache_path = os.path.join(cfg.working_dir, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to recomputing if cache is corrupt

    # 2. Compute from scratch
    # Load metadata splits
    df_train = pd.read_csv(cfg.train_path)
    df_val = pd.read_csv(cfg.val_path)

    # Concatenate to form full training set for Cross-Validation
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Load and merge CPC Contexts
    cpc_df = get_cpc_texts(cfg, load_cached_data=load_cached_data)
    df = df.merge(cpc_df, on="context", how="left")

    # Fill missing context texts if any
    df["context_text"] = df["context_text"].fillna("")

    # Create Stratified Folds
    # We stratify by 'score'. Since scores are discrete (0.0, 0.25, ...),
    # we can treat them as classes for stratification.
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    # Create a temporary column for stratification (convert float to string to ensure categorical treatment)
    df["stratify_col"] = df["score"].astype(str)

    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["stratify_col"])):
        df.loc[val_idx, "fold"] = fold

    df = df.drop(columns=["stratify_col"])

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(cfg: Config, fold: int, load_cached_data: bool = True):
    """
    Returns train and validation DataLoaders for a specific fold.
    """
    # Load processed data with folds
    df = process_data(cfg, load_cached_data=load_cached_data)

    # Split into train/val for this fold
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Debugging: subset if debug mode is on
    if cfg.debug:
        train_df = train_df.head(cfg.debug_sample_size)
        val_df = val_df.head(cfg.debug_sample_size)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Create Datasets
    train_dataset = PearsonDataset(train_df, tokenizer, cfg.max_len, is_test=False)
    val_dataset = PearsonDataset(val_df, tokenizer, cfg.max_len, is_test=False)

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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(cfg: Config, load_cached_data: bool = True):
    """
    Returns DataLoader for the test set.
    """
    # Load test metadata
    df_test = pd.read_csv(cfg.test_path)

    # Merge Contexts
    cpc_df = get_cpc_texts(cfg, load_cached_data=load_cached_data)
    df_test = df_test.merge(cpc_df, on="context", how="left")
    df_test["context_text"] = df_test["context_text"].fillna("")

    if cfg.debug:
        df_test = df_test.head(cfg.debug_sample_size)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    test_dataset = PearsonDataset(df_test, tokenizer, cfg.max_len, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
