import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from library.config import Config
from library.utils import seed_everything


class EssayDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT).
    Returns unpadded tokenized inputs and float scores.
    """

    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.text_col = Config.text_col
        self.target_col = Config.target_col
        self.id_col = Config.id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        # Tokenize without padding (dynamic padding handled by collator)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,  # DeBERTa-v3 usually doesn't need this, but safe to omit
            return_attention_mask=True,
        )

        sample = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        # Handle Targets
        if not self.is_test:
            score = row[self.target_col]
            sample["labels"] = float(score)

        # We generally don't pass strings (essay_id) to the model forward pass
        # IDs are managed via the order of the DataLoader

        return sample


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM).
    Used for Domain Adaptation.
    """

    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = Config.text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.text_col])

        # Tokenize for MLM
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "special_tokens_mask": inputs["special_tokens_mask"],
        }


class SmartCollator:
    """
    Custom collator that handles dynamic padding for inputs
    and stacking for regression labels.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padder = DataCollatorWithPadding(tokenizer, padding=True)

    def __call__(self, batch):
        # Extract labels if they exist
        labels = None
        if "labels" in batch[0]:
            labels = [item.pop("labels") for item in batch]
            labels = torch.tensor(labels, dtype=torch.float)

        # Pad inputs using HuggingFace's robust logic
        batch_out = self.padder(batch)

        # Re-attach labels
        if labels is not None:
            batch_out["labels"] = labels

        return batch_out


def load_dataframes(load_cached_data=True):
    """
    Loads Train, Validation, and Test dataframes.
    Implements Parquet caching logic.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define source and cache paths
    paths = {
        "train": (Config.train_path, Config.train_cache_path),
        "val": (Config.val_path, Config.val_cache_path),
        "test": (Config.test_path, Config.test_cache_path),
    }

    dfs = {}

    for key, (csv_path, cache_path) in paths.items():
        loaded = False

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                dfs[key] = pd.read_parquet(cache_path)
                loaded = True
            except Exception as e:
                print(
                    f"Warning: Failed to load cached {key} data: {e}. Reloading from source."
                )

        # 2. If not loaded, process from source
        if not loaded:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Source file {csv_path} not found.")

            df = pd.read_csv(csv_path)

            # Basic cleaning: Ensure text is string and handle NaNs
            if Config.text_col in df.columns:
                df[Config.text_col] = df[Config.text_col].fillna("").astype(str)

            # Save to cache
            df.to_parquet(cache_path, index=False)
            dfs[key] = df

    # Debug: Subsample if required
    if Config.debug:
        seed_everything(Config.seed)
        for key in dfs:
            dfs[key] = (
                dfs[key]
                .sample(n=min(len(dfs[key]), 100), random_state=Config.seed)
                .reset_index(drop=True)
            )

    return dfs["train"], dfs["val"], dfs["test"]


def load_mlm_dataframe(load_cached_data=True):
    """
    Creates or loads the combined corpus (Train + Val + Test) for MLM.
    """
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_path = Config.mlm_data_cache_path

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load cached MLM data: {e}. Recomputing.")

    # 2. Compute from scratch
    # Load all raw splits (ignoring individual caches to ensure integrity of combined set)
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # Combine
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Keep only text
    df_combined = df_combined[[Config.text_col]].fillna("").astype(str)

    # Save cache
    df_combined.to_parquet(cache_path, index=False)

    # Debug
    if Config.debug:
        seed_everything(Config.seed)
        df_combined = df_combined.sample(
            n=min(len(df_combined), 100), random_state=Config.seed
        ).reset_index(drop=True)

    return df_combined


def get_supervised_loaders(tokenizer, load_cached_data=True):
    """
    Constructs DataLoaders for the Supervised Fine-Tuning stage.

    Args:
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.seed)

    # Load Data
    df_train, df_val, df_test = load_dataframes(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = EssayDataset(df_train, tokenizer, Config.max_length, is_test=False)
    val_dataset = EssayDataset(df_val, tokenizer, Config.max_length, is_test=False)
    test_dataset = EssayDataset(df_test, tokenizer, Config.max_length, is_test=True)

    # Initialize Custom Collator
    collator = SmartCollator(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def get_mlm_loader(tokenizer, load_cached_data=True):
    """
    Constructs DataLoader for the Masked Language Modeling stage.

    Args:
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        mlm_loader
    """
    seed_everything(Config.seed)

    # Load Data
    df_mlm = load_mlm_dataframe(load_cached_data=load_cached_data)

    # Create Dataset
    mlm_dataset = MLMDataset(df_mlm, tokenizer, Config.max_length)

    # Collator for MLM (handles masking)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_mask_probability
    )

    # Create DataLoader
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=Config.mlm_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    return mlm_loader
