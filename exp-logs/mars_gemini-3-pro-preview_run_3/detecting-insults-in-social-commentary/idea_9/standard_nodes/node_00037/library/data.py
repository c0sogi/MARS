import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import ModelConfig
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization and tensor conversion.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.texts = df["Comment"].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not self.is_test:
            # Ensure targets are float for BCEWithLogitsLoss
            self.targets = df["Insult"].values.astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by encode_plus
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def load_processed_data(load_cached_data=True, debug=False):
    """
    Loads train, validation, and test datasets.
    Applies text decoding and caching.

    Args:
        load_cached_data (bool): Whether to try loading from parquet cache.
        debug (bool): Whether to slice data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = ModelConfig.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "train_decoded.parquet")
    val_path = os.path.join(cache_dir, "val_decoded.parquet")
    test_path = os.path.join(cache_dir, "test_decoded.parquet")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
    else:
        # Load from metadata
        train_meta = os.path.join(ModelConfig.input_dir, "train.csv")
        val_meta = os.path.join(ModelConfig.input_dir, "val.csv")
        test_meta = os.path.join(ModelConfig.input_dir, "test.csv")

        train_df = pd.read_csv(train_meta)
        val_df = pd.read_csv(val_meta)
        test_df = pd.read_csv(test_meta)

        # Decode text
        train_df["Comment"] = train_df["Comment"].apply(decode_text)
        val_df["Comment"] = val_df["Comment"].apply(decode_text)
        test_df["Comment"] = test_df["Comment"].apply(decode_text)

        # Save to cache
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

    # Apply debug slicing if requested
    if debug or ModelConfig.debug:
        sample_size = ModelConfig.debug_sample_size
        train_df = train_df.iloc[:sample_size]
        val_df = val_df.iloc[:sample_size]
        test_df = test_df.iloc[:sample_size]

    return train_df, val_df, test_df


def create_augmented_dataset(train_df, test_df, test_probs):
    """
    Merges original training data with high-confidence pseudo-labeled test data.

    Args:
        train_df (pd.DataFrame): Original training data.
        test_df (pd.DataFrame): Test data (features).
        test_probs (np.array): Predicted probabilities for test data.

    Returns:
        pd.DataFrame: Augmented training dataset.
    """
    # Copy test dataframe to avoid mutation
    test_aug = test_df.copy()
    test_aug["prob"] = test_probs

    # Identify high confidence samples
    # Class 1 (Insult)
    high_conf_mask = test_aug["prob"] >= ModelConfig.pseudo_label_threshold_high
    # Class 0 (Neutral)
    low_conf_mask = test_aug["prob"] <= ModelConfig.pseudo_label_threshold_low

    # Assign pseudo-labels
    test_aug.loc[high_conf_mask, "Insult"] = 1
    test_aug.loc[low_conf_mask, "Insult"] = 0

    # Filter only confident samples
    pseudo_labeled_df = test_aug[high_conf_mask | low_conf_mask].copy()

    # Clean up columns to match training data
    pseudo_labeled_df = pseudo_labeled_df.drop(columns=["prob"])
    pseudo_labeled_df["Insult"] = pseudo_labeled_df["Insult"].astype(int)

    # Ensure column order matches
    pseudo_labeled_df = pseudo_labeled_df[train_df.columns]

    # Merge
    augmented_df = pd.concat([train_df, pseudo_labeled_df], axis=0).reset_index(
        drop=True
    )

    return augmented_df


def get_dataloaders(train_df, val_df, test_df, tokenizer_name):
    """
    Creates DataLoaders for the given dataframes and tokenizer.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        tokenizer_name (str): HuggingFace tokenizer name.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create Datasets
    train_dataset = InsultDataset(
        train_df, tokenizer, ModelConfig.max_length, is_test=False
    )

    val_dataset = InsultDataset(
        val_df, tokenizer, ModelConfig.max_length, is_test=False
    )

    test_dataset = InsultDataset(
        test_df, tokenizer, ModelConfig.max_length, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConfig.train_batch_size,
        shuffle=True,
        num_workers=ModelConfig.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.valid_batch_size,
        shuffle=False,
        num_workers=ModelConfig.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=ModelConfig.valid_batch_size,
        shuffle=False,
        num_workers=ModelConfig.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
