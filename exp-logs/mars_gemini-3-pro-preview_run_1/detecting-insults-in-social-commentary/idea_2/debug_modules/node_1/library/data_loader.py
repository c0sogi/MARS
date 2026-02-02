import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


def preprocess_text(text):
    """
    Preprocesses text for the Structural (NBSVM) branch.
    1. Unescapes unicode sequences (e.g., \\n -> \n, \\xe2 -> char).
    2. Lowercases the text.
    3. Strips non-ASCII unicode characters.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Attempt to decode unicode escapes if they are present as literal characters
    try:
        # Encode to utf-8 bytes then decode as unicode_escape to resolve literals like \\n or \\xe2
        text = text.encode("utf-8").decode("unicode_escape")
    except Exception:
        # If decoding fails, proceed with the original text
        pass

    # Lowercase
    text = text.lower()

    # Strip unicode characters (keep only ASCII)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def load_data(load_cached_data=True, debug_size=None):
    """
    Loads training, validation, and test data.
    Implements caching using Parquet files in the working directory.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_size (int, optional): Number of samples to return for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # Determine if we should load from cache
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Loading data from metadata and processing...")
        # Load raw data from metadata
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply preprocessing for the Structural Branch (NBSVM)
        # We keep the original 'Comment' for the Semantic Branch (Transformer)
        print("Preprocessing text for NBSVM branch...")
        train_df["clean_comment"] = train_df["Comment"].apply(preprocess_text)
        val_df["clean_comment"] = val_df["Comment"].apply(preprocess_text)
        test_df["clean_comment"] = test_df["Comment"].apply(preprocess_text)

        # Save to cache
        print("Saving processed data to cache...")
        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)
        test_df.to_parquet(test_cache)

    # Apply debugging limit if specified
    # If Config.DEBUG is True and debug_size is not provided, use Config.DEBUG_SAMPLE_SIZE
    if debug_size is None and Config.DEBUG:
        debug_size = Config.DEBUG_SAMPLE_SIZE

    if debug_size is not None:
        print(f"Debug mode: Slicing datasets to {debug_size} samples.")
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    return train_df, val_df, test_df


class InsultDataset(Dataset):
    """
    PyTorch Dataset for the Semantic Branch (Transformer).
    Tokenizes text using the provided HuggingFace tokenizer.
    """

    def __init__(self, texts, tokenizer, labels=None, max_len=Config.MAX_LEN):
        """
        Args:
            texts (list or pd.Series): List of text comments.
            tokenizer: HuggingFace tokenizer instance.
            labels (list or pd.Series, optional): List of target labels (0 or 1).
            max_len (int): Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # token_type_ids are usually not needed for RoBERTa but returned for compatibility

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

        if self.labels is not None:
            # Labels need to be float for BCEWithLogitsLoss or Long for CrossEntropy
            # Assuming binary classification with a single output neuron (BCE)
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
