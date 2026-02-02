import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    TARGET_COLS,
    SEED,
)

# Prevent tokenizer parallelism issues with DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for StackExchange Question-Answer pairs.
    Prepares inputs for a Siamese network:
    - Input A: Question Title + [SEP] + Question Body
    - Input B: Answer
    """

    def __init__(self, df, tokenizer, max_length, target_cols=None, is_test=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_cols = target_cols
        self.is_test = is_test

        # Pre-extract data to lists/arrays for faster access in __getitem__
        self.qa_ids = df["qa_id"].values
        self.titles = df["question_title"].fillna("").astype(str).tolist()
        self.bodies = df["question_body"].fillna("").astype(str).tolist()
        self.answers = df["answer"].fillna("").astype(str).tolist()

        if not self.is_test and self.target_cols:
            self.labels = df[self.target_cols].values.astype("float32")
        else:
            self.labels = None

    def __len__(self):
        return len(self.qa_ids)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Construct Question string: Title + Separator + Body
        # We use the tokenizer's separator if available, otherwise a space
        sep = self.tokenizer.sep_token if self.tokenizer.sep_token else " "
        question_text = f"{title} {sep} {body}"

        # Tokenize Question
        q_enc = self.tokenizer(
            question_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize Answer
        a_enc = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare output dictionary
        item = {
            "qa_id": self.qa_ids[idx],
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_attention_mask": q_enc["attention_mask"].squeeze(0),
            "a_input_ids": a_enc["input_ids"].squeeze(0),
            "a_attention_mask": a_enc["attention_mask"].squeeze(0),
        }

        if not self.is_test and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def load_processed_data(load_cached_data=True):
    """
    Loads data from metadata CSVs. Implements caching using Parquet.
    """
    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_processed.parquet")

    def _load_or_process(source_path, cache_path, is_test=False):
        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Process from scratch
        print(f"Processing raw data from {source_path}")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        df = pd.read_csv(source_path)

        # Basic cleaning: Ensure text columns are strings
        text_cols = ["question_title", "question_body", "answer"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Save to cache
        print(f"Saving processed data to {cache_path}")
        df.to_parquet(cache_path, index=False)
        return df

    # Load all splits
    df_train = _load_or_process(TRAIN_METADATA_PATH, train_cache)
    df_val = _load_or_process(VAL_METADATA_PATH, val_cache)
    df_test = _load_or_process(TEST_METADATA_PATH, test_cache, is_test=True)

    return df_train, df_val, df_test


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to get DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        debug (bool): If True, subsets data to a small amount for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Data
    df_train, df_val, df_test = load_processed_data(load_cached_data=load_cached_data)

    # 2. Handle Debug Mode
    if debug:
        print("DEBUG MODE: Subsetting data to 100 rows per split.")
        df_train = df_train.head(100)
        df_val = df_val.head(100)
        df_test = df_test.head(100)

    # 3. Initialize Tokenizer
    print(f"Initializing tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Create Datasets
    print("Creating Datasets...")
    train_dataset = StackExchangeDataset(
        df_train, tokenizer, MAX_LENGTH, target_cols=TARGET_COLS, is_test=False
    )
    val_dataset = StackExchangeDataset(
        df_val, tokenizer, MAX_LENGTH, target_cols=TARGET_COLS, is_test=False
    )
    test_dataset = StackExchangeDataset(
        df_test, tokenizer, MAX_LENGTH, target_cols=None, is_test=True
    )

    # 5. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders ready. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches."
    )
    return train_loader, val_loader, test_loader
