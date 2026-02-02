import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from library.config import Config
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization and formatting for DeBERTa models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        is_test: bool = False,
    ):
        self.texts = df["Comment"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["Insult"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            # HuggingFace models typically expect LongTensor for classification labels
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_and_preprocess_data(load_cached_data: bool = True):
    """
    Loads data from CSVs, applies decoding, and handles caching via Parquet files.
    Strictly follows the required caching logic.
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    train_cache = os.path.join(Config.cache_dir, "train_decoded.parquet")
    val_cache = os.path.join(Config.cache_dir, "val_decoded.parquet")
    test_cache = os.path.join(Config.cache_dir, "test_decoded.parquet")

    # Logic Flow: 1. Try to load cached
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception:
                # If loading fails, proceed to process from scratch
                pass

    # Logic Flow: 2. Process from scratch if cache miss or load_cached_data is False
    # Load raw CSVs
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Apply deterministic processing (decoding)
    train_df["Comment"] = train_df["Comment"].apply(decode_text)
    val_df["Comment"] = val_df["Comment"].apply(decode_text)
    test_df["Comment"] = test_df["Comment"].apply(decode_text)

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def prepare_augmented_data(train_df: pd.DataFrame, test_df: pd.DataFrame, test_probs):
    """
    Creates an augmented training dataset by adding pseudo-labeled test samples.

    Args:
        train_df: Original training DataFrame.
        test_df: Test DataFrame.
        test_probs: Array-like of probabilities for the test set (class 1 probability).

    Returns:
        augmented_df: Combined DataFrame of original train + pseudo-labeled test.
    """
    # Create a copy to avoid modifying the original test_df
    pseudo_df = test_df.copy()

    # Assign probabilities
    pseudo_df["probability"] = test_probs

    # Filter for high confidence samples
    # Class 1: prob >= threshold
    # Class 0: prob <= (1 - threshold)
    threshold = Config.pseudo_label_threshold

    high_conf_1 = pseudo_df[pseudo_df["probability"] >= threshold].copy()
    high_conf_1["Insult"] = 1

    high_conf_0 = pseudo_df[pseudo_df["probability"] <= (1.0 - threshold)].copy()
    high_conf_0["Insult"] = 0

    # Combine pseudo-labeled data
    pseudo_labeled = pd.concat([high_conf_1, high_conf_0], ignore_index=True)

    # Drop the probability column to match train_df structure
    pseudo_labeled = pseudo_labeled.drop(columns=["probability"])

    # Concatenate with original training data
    augmented_df = pd.concat([train_df, pseudo_labeled], ignore_index=True)

    # Shuffle the augmented dataset
    augmented_df = augmented_df.sample(frac=1, random_state=Config.seed).reset_index(
        drop=True
    )

    return augmented_df


def create_dataloaders(
    tokenizer, train_df=None, val_df=None, test_df=None, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.
    If DataFrames are not provided, they are loaded using load_and_preprocess_data.
    """
    # Load data if not provided
    if train_df is None or val_df is None or test_df is None:
        loaded_train, loaded_val, loaded_test = load_and_preprocess_data(
            load_cached_data=load_cached_data
        )
        train_df = train_df if train_df is not None else loaded_train
        val_df = val_df if val_df is not None else loaded_val
        test_df = test_df if test_df is not None else loaded_test

    # Handle Debug Mode
    if Config.debug:
        train_df = train_df.iloc[: Config.debug_subset_size]
        val_df = val_df.iloc[: Config.debug_subset_size]
        test_df = test_df.iloc[: Config.debug_subset_size]

    # Create Datasets
    train_dataset = InsultDataset(train_df, tokenizer, Config.max_length, is_test=False)
    val_dataset = InsultDataset(val_df, tokenizer, Config.max_length, is_test=False)
    test_dataset = InsultDataset(test_df, tokenizer, Config.max_length, is_test=True)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
