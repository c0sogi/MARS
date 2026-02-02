import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything

# ====================================================
# Dataset Classes
# ====================================================


class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False, label_cols=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values
        self.label_cols = label_cols if label_cols else Config.target_cols

        if not self.is_test:
            self.labels = df[self.label_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        item = {"input_ids": ids, "attention_mask": mask}

        if not self.is_test:
            # Returns float tensors to support both binary and soft labels
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            item["labels"] = labels

        return item


class MLMDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df["comment_text"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            # DataCollatorForLanguageModeling will handle masking and labels
        }


# ====================================================
# Data Loading & Caching Logic
# ====================================================


def load_data(load_cached_data=True):
    """
    Loads data from metadata and raw files.
    Implements caching using Parquet to speed up subsequent runs.
    """
    os.makedirs(Config.cache_dir, exist_ok=True)

    train_cache_path = os.path.join(Config.cache_dir, "train_cache.parquet")
    test_cache_path = os.path.join(Config.cache_dir, "test_cache.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        # print(f"Loading data from cache: {Config.cache_dir}")
        train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        return train_df, test_df

    # 2. If not cached, process from scratch
    # print("Cache miss or reload requested. Processing raw data...")

    # Load Metadata
    train_meta = pd.read_csv(Config.train_metadata_path)
    val_meta = pd.read_csv(Config.val_metadata_path)
    test_meta = pd.read_csv(Config.test_metadata_path)

    # Combine train and val metadata to get the full training set for CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Raw Data
    # We assume raw files are in Config.input_dir
    raw_train = pd.read_csv(Config.train_raw_path)
    raw_test = pd.read_csv(Config.test_raw_path)

    # Merge Metadata with Raw Text
    # full_train_meta has columns: id, labels..., source_file
    # raw_train has columns: id, comment_text, ...
    train_df = pd.merge(
        full_train_meta, raw_train[["id", "comment_text"]], on="id", how="left"
    )
    test_df = pd.merge(test_meta, raw_test[["id", "comment_text"]], on="id", how="left")

    # Fill missing text
    train_df["comment_text"] = train_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    # Save to cache
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df


# ====================================================
# Loader Generation Functions
# ====================================================


def get_dapt_loaders(tokenizer, load_cached_data=True):
    """
    Creates a DataLoader for Domain-Adaptive Pre-training (MLM).
    Combines Train and Test text.
    """
    train_df, test_df = load_data(load_cached_data)

    # Combine all text for DAPT
    dapt_df = pd.concat(
        [train_df[["comment_text"]], test_df[["comment_text"]]], ignore_index=True
    )

    if Config.debug:
        dapt_df = dapt_df.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)

    dataset = MLMDataset(dapt_df, tokenizer, Config.max_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.dapt_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    return loader


def get_teacher_loaders(fold, tokenizer, load_cached_data=True):
    """
    Creates Train and Val DataLoaders for a specific fold for Teacher training.
    Uses StratifiedKFold on the full training set.
    """
    train_df, _ = load_data(load_cached_data)

    if Config.debug:
        train_df = train_df.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)

    # Create Stratified Folds
    # We stratify based on the 'toxic' column as a proxy for distribution
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # We need a target for stratification.
    # Using 'toxic' is a common heuristic if full multi-label stratification isn't available.
    stratify_target = train_df["toxic"]

    # Get indices for the requested fold
    for f, (train_idx, val_idx) in enumerate(skf.split(train_df, stratify_target)):
        if f == fold:
            train_fold = train_df.iloc[train_idx].reset_index(drop=True)
            val_fold = train_df.iloc[val_idx].reset_index(drop=True)
            break

    train_dataset = ToxicDataset(train_fold, tokenizer, Config.max_len)
    val_dataset = ToxicDataset(val_fold, tokenizer, Config.max_len)

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
        drop_last=False,
    )

    return train_loader, val_loader


def get_student_loaders(tokenizer, test_pseudo_labels_df=None, load_cached_data=True):
    """
    Creates a DataLoader for Student training.
    Combines original Train data (hard labels) with Test data (soft pseudo-labels).

    Args:
        test_pseudo_labels_df: DataFrame containing 'id' and soft label columns for the test set.
                               If None, only the training data is used (not recommended for student stage).
    """
    train_df, test_df = load_data(load_cached_data)

    if Config.debug:
        train_df = train_df.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        if test_pseudo_labels_df is not None:
            # Filter pseudo labels to match debug subset if we were to sample test,
            # but usually we want all test data. For debug, let's sample test too.
            test_df = test_df.sample(
                n=Config.debug_sample_size, random_state=Config.seed
            ).reset_index(drop=True)
            test_pseudo_labels_df = test_pseudo_labels_df[
                test_pseudo_labels_df["id"].isin(test_df["id"])
            ]

    # Prepare Train Data (Hard Labels -> Float)
    # train_df already has binary labels in target_cols

    # Prepare Test Data (Soft Labels)
    if test_pseudo_labels_df is not None:
        # Merge soft labels into test_df
        # Ensure we don't have label columns in test_df before merging
        cols_to_drop = [c for c in Config.target_cols if c in test_df.columns]
        test_df_clean = test_df.drop(columns=cols_to_drop)

        # Merge on ID
        test_labeled = pd.merge(
            test_df_clean, test_pseudo_labels_df, on="id", how="inner"
        )

        # Combine
        combined_df = pd.concat([train_df, test_labeled], ignore_index=True)
    else:
        combined_df = train_df

    dataset = ToxicDataset(combined_df, tokenizer, Config.max_len)

    loader = DataLoader(
        dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def get_test_loader(tokenizer, load_cached_data=True):
    """
    Creates a DataLoader for the Test set (Inference).
    """
    _, test_df = load_data(load_cached_data)

    dataset = ToxicDataset(test_df, tokenizer, Config.max_len, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
