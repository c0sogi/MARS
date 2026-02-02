import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


def load_data(split: str, load_cached_data: bool = True):
    """
    Loads data from metadata CSVs, performs basic cleaning, and implements caching via Parquet.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded and cleaned dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}.parquet")

    # map split to source path
    if split == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        source_path = Config.VAL_DATA_PATH
    elif split == "test":
        source_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(
                f"Failed to load cache for {split}: {e}. Reloading from source."
            )

    # 2. Load from source and process
    logger.info(f"Loading {split} data from source: {source_path}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Basic cleaning: Fill NaNs in text columns to avoid tokenization errors
    text_cols = ["prompt", "response_a", "response_b"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 3. Save to cache
    logger.info(f"Saving {split} data to cache: {cache_path}")
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        logger.warning(f"Failed to save cache for {split}: {e}")

    return df


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Dual-Encoder.
    Tokenizes (Prompt, Response A) and (Prompt, Response B) separately.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-compute label indices if not test
        if not self.is_test:
            # 0: Model A, 1: Model B, 2: Tie
            self.labels = []
            # We assume the dataframe has the one-hot columns
            # Using numpy argmax is efficient
            cols = ["winner_model_a", "winner_model_b", "winner_tie"]
            if all(c in df.columns for c in cols):
                self.labels = np.argmax(df[cols].values, axis=1)
            else:
                raise ValueError("Training data missing target columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        prompt = row["prompt"]
        resp_a = row["response_a"]
        resp_b = row["response_b"]

        # Tokenize (Prompt, Response A)
        # The tokenizer handles [CLS] Prompt [SEP] Response A [SEP] structure automatically
        enc_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize (Prompt, Response B)
        enc_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids_a": enc_a["input_ids"].squeeze(0),  # Remove batch dim
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # For test set, pass ID to help with submission file creation if needed
            # (Though usually prediction loop handles IDs separately, it's good to have)
            item["id"] = row["id"]

        return item


def get_dataloaders(
    tokenizer_name=Config.MODEL_NAME,
    train_batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
    max_length=Config.MAX_LENGTH,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    logger.info(f"Initializing Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load DataFrames
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Handle Debug Mode
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        logger.info(
            f"DEBUG mode enabled. Truncating datasets to {subset_size} samples."
        )
        train_df = train_df.iloc[:subset_size]
        val_df = val_df.iloc[:subset_size]
        # We usually want to predict on full test even in debug, or maybe subset too.
        # For safety in a pipeline run, we'll subset test too if debug is on.
        test_df = test_df.iloc[:subset_size]

    logger.info(
        f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Create Datasets
    train_dataset = ChatbotDataset(train_df, tokenizer, max_length, is_test=False)
    val_dataset = ChatbotDataset(val_df, tokenizer, max_length, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, max_length, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=valid_batch_size,  # Use valid batch size for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
