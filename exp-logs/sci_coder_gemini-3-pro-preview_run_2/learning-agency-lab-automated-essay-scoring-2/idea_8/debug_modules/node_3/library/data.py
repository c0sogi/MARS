import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import CFG
from library.utils import seed_everything


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Wraps pre-tokenized numpy arrays to minimize overhead.
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
            # Regression target: float
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def _load_and_process(
    file_paths, cache_name, tokenizer, max_length, load_cached_data=True
):
    """
    Loads CSVs, tokenizes text, and manages .npy caching.
    """
    # Ensure cache directory exists
    cache_dir = os.path.join(CFG.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{cache_name}_input_ids.npy")
    mask_path = os.path.join(cache_dir, f"{cache_name}_attention_mask.npy")
    labels_path = os.path.join(cache_dir, f"{cache_name}_labels.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(ids_path) and os.path.exists(mask_path):
        input_ids = np.load(ids_path)
        attention_mask = np.load(mask_path)

        # Load labels if they exist (train/val sets)
        labels = None
        if os.path.exists(labels_path):
            labels = np.load(labels_path)

        return input_ids, attention_mask, labels

    # --- Processing from scratch ---

    # Load and concat all provided CSVs
    dfs = []
    for p in file_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            raise FileNotFoundError(f"File not found: {p}")

    df = pd.concat(dfs, ignore_index=True)

    # Handle missing text
    df["full_text"] = df["full_text"].fillna("").astype(str)

    # Tokenize
    # Using batch_encode_plus is significantly faster than iterating
    encoded = tokenizer.batch_encode_plus(
        df["full_text"].tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",  # Return numpy arrays directly
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Handle Labels
    labels = None
    if "score" in df.columns:
        labels = df["score"].values.astype(np.float32)
        np.save(labels_path, labels)

    # Save features to cache
    np.save(ids_path, input_ids)
    np.save(mask_path, attention_mask)

    return input_ids, attention_mask, labels


def get_loaders(fold, load_cached_data=True):
    """
    Creates Train and Validation DataLoaders for a specific fold.
    Merges metadata train and val files to perform Stratified 5-Fold CV.
    """
    seed_everything(CFG.seed)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Merge train and val metadata to ensure we have the full dataset for CV
    input_ids, attention_mask, labels = _load_and_process(
        [CFG.train_path, CFG.val_path],
        "train_val_merged",
        tokenizer,
        CFG.max_length,
        load_cached_data,
    )

    if labels is None:
        raise ValueError("Labels not found in training data.")

    # Stratified K-Fold Split
    # We cast labels to int for stratification (assuming scores 1-6)
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

    # Create dummy X for the split method
    X_dummy = np.zeros(len(labels))

    # Get indices for the requested fold
    splits = list(skf.split(X_dummy, labels.astype(int)))
    train_idx, val_idx = splits[fold]

    # Create Datasets
    train_ds = EssayDataset(
        input_ids[train_idx], attention_mask[train_idx], labels[train_idx]
    )

    val_ds = EssayDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates a DataLoader for the test set.
    Returns the loader and the essay_ids for submission mapping.
    """
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    input_ids, attention_mask, _ = _load_and_process(
        [CFG.test_path], "test", tokenizer, CFG.max_length, load_cached_data
    )

    test_ds = EssayDataset(input_ids, attention_mask, labels=None)

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Return essay_ids to align predictions later
    essay_ids = pd.read_csv(CFG.test_path)["essay_id"].values

    return test_loader, essay_ids
