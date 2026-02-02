import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import clean_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization and input formatting for DeBERTa.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["Comment"].values
        # Handle target if available
        if not is_test:
            self.targets = df["Insult"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            inputs["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return inputs


def load_train_data(load_cached_data=True):
    """
    Loads training and validation metadata, combines them, cleans text,
    and returns a single dataframe. Implements caching to parquet.
    """
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_path = os.path.join(Config.working_dir, "full_train_cleaned.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load metadata
    df_train_meta = pd.read_csv(Config.train_path)
    df_val_meta = pd.read_csv(Config.val_path)

    # Combine for CV
    df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Clean text
    df["Comment"] = df["Comment"].apply(clean_text)

    # Cache
    df.to_parquet(cache_path, index=False)

    return df


def load_test_data(load_cached_data=True):
    """
    Loads test metadata, cleans text, and returns dataframe.
    Implements caching.
    """
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_path = os.path.join(Config.working_dir, "test_cleaned.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load metadata
    df = pd.read_csv(Config.test_path)

    # Clean text
    df["Comment"] = df["Comment"].apply(clean_text)

    # Cache
    df.to_parquet(cache_path, index=False)

    return df


def get_folds(df, n_folds, seed):
    """
    Adds a 'fold' column to the dataframe using StratifiedKFold.
    """
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1

    # Stratify by Insult label
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Insult"])):
        df.loc[val_idx, "fold"] = fold

    return df


def prepare_loaders(fold, df=None, load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for a specific fold.

    Args:
        fold (int): The fold index to use for validation.
        df (pd.DataFrame, optional): If provided, uses this dataframe (e.g., for pseudo-labeling).
                                     Otherwise loads default train data.
        load_cached_data (bool): Whether to use cached data.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, valid_loader
    """
    # Load data if not provided
    if df is None:
        df = load_train_data(load_cached_data=load_cached_data)

    # Debug mode
    if debug:
        df = df.sample(
            n=min(len(df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # Assign folds
    df = get_folds(df, n_folds=Config.n_folds, seed=Config.seed)

    # Split
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Datasets
    train_dataset = InsultDataset(train_df, tokenizer, Config.max_len, is_test=False)
    valid_dataset = InsultDataset(valid_df, tokenizer, Config.max_len, is_test=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def prepare_test_loader(load_cached_data=True, debug=False):
    """
    Prepares DataLoader for the test set.
    """
    df = load_test_data(load_cached_data=load_cached_data)

    if debug:
        df = df.sample(
            n=min(len(df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    dataset = InsultDataset(df, tokenizer, Config.max_len, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def merge_pseudo_labels(train_df, test_df, preds, threshold=0.90):
    """
    Merges high-confidence test predictions into the training set.

    Args:
        train_df (pd.DataFrame): Original training dataframe.
        test_df (pd.DataFrame): Test dataframe (cleaned).
        preds (np.array): Predicted probabilities for the test set.
        threshold (float): Confidence threshold for pseudo-labeling.

    Returns:
        pd.DataFrame: Augmented dataframe.
    """
    # Create a copy to avoid modifying original
    pseudo_df = test_df.copy()

    # Identify high confidence samples
    # Class 1: prob > threshold
    # Class 0: prob < (1 - threshold)
    idx_1 = np.where(preds > threshold)[0]
    idx_0 = np.where(preds < (1 - threshold))[0]

    # Assign labels
    pseudo_df.loc[idx_1, "Insult"] = 1
    pseudo_df.loc[idx_0, "Insult"] = 0

    # Filter only labeled samples
    labeled_indices = np.concatenate([idx_1, idx_0])
    pseudo_df = pseudo_df.iloc[labeled_indices].copy()

    # Ensure types match
    pseudo_df["Insult"] = pseudo_df["Insult"].astype(int)

    # Select relevant columns
    cols = ["Insult", "Date", "Comment"]
    # Ensure columns exist (Date might be in test_df)
    available_cols = [c for c in cols if c in pseudo_df.columns]
    pseudo_df = pseudo_df[available_cols]

    # Concatenate
    augmented_df = pd.concat([train_df[available_cols], pseudo_df], ignore_index=True)

    return augmented_df
