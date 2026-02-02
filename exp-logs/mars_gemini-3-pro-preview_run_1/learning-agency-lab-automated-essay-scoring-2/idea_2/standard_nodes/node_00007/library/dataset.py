import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class EssayDataset(Dataset):
    """
    Custom Dataset for Essay Scoring.
    Handles tokenization and preparation of inputs for DeBERTa.
    """

    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = df["full_text"].values

        if not self.is_test:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the essay text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by return_tensors='pt'
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            # Regression target: float tensor
            score = torch.tensor(self.scores[idx], dtype=torch.float)
            item["labels"] = score

        return item


def load_data(load_cached_data=True):
    """
    Loads datasets from CSV or Parquet cache.
    Implements caching mechanism to save processed DataFrames.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    datasets = {}
    files_map = {
        "train": (Config.train_path, Config.train_cache_path),
        "val": (Config.val_path, Config.val_cache_path),
        "test": (Config.test_path, Config.test_cache_path),
    }

    for key, (csv_path, cache_path) in files_map.items():
        df = None

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                try:
                    df = pd.read_parquet(cache_path)
                except Exception:
                    pass

        # 2. If not loaded, load from source and save to cache
        if df is None:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Cache the dataframe
                df.to_parquet(cache_path, index=False)
            else:
                raise FileNotFoundError(f"Source file {csv_path} not found.")

        datasets[key] = df

    return datasets["train"], datasets["val"], datasets["test"]


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Constructs DataLoaders for train, val, and test sets.

    Args:
        tokenizer: Pre-trained tokenizer instance.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Debugging: Use a small subset of data
    if Config.debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # Instantiate Datasets
    train_dataset = EssayDataset(
        train_df, tokenizer, max_length=Config.max_length, is_test=False
    )
    val_dataset = EssayDataset(
        val_df, tokenizer, max_length=Config.max_length, is_test=False
    )
    test_dataset = EssayDataset(
        test_df, tokenizer, max_length=Config.max_length, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
