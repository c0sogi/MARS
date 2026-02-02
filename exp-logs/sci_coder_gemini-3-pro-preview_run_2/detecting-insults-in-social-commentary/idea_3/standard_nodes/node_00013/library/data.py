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
    def __init__(self, df, inference_only=False):
        self.df = df
        self.inference_only = inference_only

        # Expecting input_ids and attention_mask to be columns in the dataframe
        # containing lists of integers
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()

        if not self.inference_only:
            self.labels = df["Insult"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if not self.inference_only:
            # Float for BCEWithLogitsLoss
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads data from metadata, cleans text, tokenizes, and creates Stratified K-Folds.
    Implements caching using Parquet files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (train_df, test_df) with tokenized columns and fold indices.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    train_cache_path = os.path.join(Config.working_dir, "train_tokens.parquet")
    test_cache_path = os.path.join(Config.working_dir, "test_tokens.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached data from {Config.working_dir}...")
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)

            if "fold" not in train_df.columns:
                raise ValueError("Cached data missing 'fold' column")

            # Parquet might load lists as numpy arrays, ensure compatibility if needed
            # but usually pandas handles object columns of lists fine.
            return train_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data.")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load and combine metadata train/val for full Cross-Validation
    df_train_part = pd.read_csv(Config.train_path)
    df_val_part = pd.read_csv(Config.val_path)
    train_df = pd.concat([df_train_part, df_val_part], ignore_index=True)

    test_df = pd.read_csv(Config.test_path)

    # Clean text
    print("Cleaning text...")
    train_df["Comment"] = train_df["Comment"].apply(clean_text)
    test_df["Comment"] = test_df["Comment"].apply(clean_text)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    def tokenize_batch(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=Config.max_len,
            return_tensors=None,  # Return python lists, easier for parquet storage
        )

    print("Tokenizing training data...")
    train_encodings = tokenize_batch(train_df["Comment"].tolist())
    train_df["input_ids"] = train_encodings["input_ids"]
    train_df["attention_mask"] = train_encodings["attention_mask"]

    print("Tokenizing test data...")
    test_encodings = tokenize_batch(test_df["Comment"].tolist())
    test_df["input_ids"] = test_encodings["input_ids"]
    test_df["attention_mask"] = test_encodings["attention_mask"]

    # Create Stratified K-Folds
    print(f"Creating {Config.n_folds}-fold Stratified Split...")
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["Insult"])):
        train_df.loc[val_idx, "fold"] = fold

    # Save to cache
    print(f"Saving processed data to {Config.working_dir}...")
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df


def get_dataloaders(train_df, fold_idx):
    """
    Returns training and validation DataLoaders for a specific fold.

    Args:
        train_df (pd.DataFrame): The full training dataframe with 'fold' column.
        fold_idx (int): The fold index to use for validation.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split based on fold
    df_train = train_df[train_df["fold"] != fold_idx].reset_index(drop=True)
    df_val = train_df[train_df["fold"] == fold_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = InsultDataset(df_train, inference_only=False)
    val_dataset = InsultDataset(df_val, inference_only=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(test_df):
    """
    Returns a DataLoader for the test set.

    Args:
        test_df (pd.DataFrame): The test dataframe with tokenized columns.

    Returns:
        DataLoader: Test data loader.
    """
    test_dataset = InsultDataset(test_df, inference_only=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader
