import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def get_tokenizer(model_name):
    """
    Instantiates the tokenizer for the given model name.

    Args:
        model_name (str): Name of the model (e.g., 'microsoft/deberta-v3-large').

    Returns:
        transformers.PreTrainedTokenizerFast: The tokenizer.
    """
    # We enforce use_fast=True to ensure sequence_ids() is available and efficient
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def load_data_with_cache(path, cache_name, load_cached_data=True):
    """
    Loads data from CSV or cached Parquet file.

    Args:
        path (str): Path to the original CSV file.
        cache_name (str): Name for the cache file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"[Data Loader] Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[Data Loader] Loading raw data from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Save to cache for future runs
    print(f"[Data Loader] Caching data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for StackExchange Question-Answer pairs.
    Handles concatenation of Question Title + Body and Answer,
    and generates segment-specific masks using tokenizer sequence_ids.
    """

    def __init__(self, df, tokenizer, max_len=512, target_cols=None, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_cols = target_cols
        self.is_test = is_test

        # Pre-extract columns to lists for efficiency
        # Fill NaNs with empty strings to avoid tokenization errors
        self.titles = df["question_title"].fillna("").astype(str).tolist()
        self.bodies = df["question_body"].fillna("").astype(str).tolist()
        self.answers = df["answer"].fillna("").astype(str).tolist()
        self.qa_ids = df["qa_id"].values

        if not self.is_test and self.target_cols:
            self.targets = df[self.target_cols].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Construct inputs
        # Segment A: Question (Title + Body)
        # Segment B: Answer
        question_text = title + " " + body
        answer_text = answer

        # Tokenize pair
        # padding='max_length' ensures consistent tensor shapes
        # truncation=True defaults to 'longest_first' strategy
        encoding = self.tokenizer(
            question_text,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension (1, seq_len) -> (seq_len)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Generate Segment Masks using sequence_ids
        # sequence_ids returns: None (special tokens), 0 (seq A), 1 (seq B)
        raw_seq_ids = encoding.sequence_ids(0)

        # Create binary masks
        # q_mask: 1 for Question tokens (seq_id == 0)
        # a_mask: 1 for Answer tokens (seq_id == 1)
        # Special tokens and padding (None) get 0 in both specific masks
        q_mask = [1 if s == 0 else 0 for s in raw_seq_ids]
        a_mask = [1 if s == 1 else 0 for s in raw_seq_ids]

        q_mask_tensor = torch.tensor(q_mask, dtype=torch.long)
        a_mask_tensor = torch.tensor(a_mask, dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "q_mask": q_mask_tensor,
            "a_mask": a_mask_tensor,
            "qa_id": self.qa_ids[idx],
        }

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(
    tokenizer,
    train_batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
    load_cached_data=True,
    debug=False,
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        tokenizer: The tokenizer instance to use.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation/testing.
        load_cached_data (bool): Whether to use cached dataframes.
        debug (bool): If True, subsets data for rapid debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = load_data_with_cache(
        Config.TRAIN_META_PATH, "train_meta", load_cached_data
    )
    val_df = load_data_with_cache(Config.VAL_META_PATH, "val_meta", load_cached_data)
    test_df = load_data_with_cache(Config.TEST_META_PATH, "test_meta", load_cached_data)

    # Debug Mode: Subset data
    if debug:
        print("[Data Loader] Debug mode enabled. Subsetting data.")
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # Instantiate Datasets
    train_dataset = StackExchangeDataset(
        train_df,
        tokenizer,
        max_len=Config.MAX_LEN,
        target_cols=Config.TARGET_COLS,
        is_test=False,
    )

    val_dataset = StackExchangeDataset(
        val_df,
        tokenizer,
        max_len=Config.MAX_LEN,
        target_cols=Config.TARGET_COLS,
        is_test=False,
    )

    test_dataset = StackExchangeDataset(
        test_df, tokenizer, max_len=Config.MAX_LEN, target_cols=None, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
