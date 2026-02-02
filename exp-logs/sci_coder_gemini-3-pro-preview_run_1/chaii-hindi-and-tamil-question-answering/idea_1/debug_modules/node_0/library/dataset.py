import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    Custom Dataset for Question Answering using mT5.
    Formats input as: "question: <question> context: <context>"
    """

    def __init__(self, data, tokenizer, source_max_len, target_max_len, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Input formatting for T5/mT5
        source_text = f"question: {row['question']} context: {row['context']}"

        # Tokenize inputs
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Base item dictionary
        item = {
            "ids": row["id"],
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "context": row["context"],  # Needed for post-processing
            "question": row["question"],  # Kept for reference
        }

        # Handle targets (Training/Validation)
        if not self.is_test:
            target_text = str(row["answer_text"])

            # Tokenize targets
            # Note: text_target is the standard way to tokenize labels in newer transformers
            target_encoding = self.tokenizer(
                text_target=target_text,
                max_length=self.target_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = target_encoding["input_ids"].squeeze()

            # Replace padding token id with -100 so CrossEntropyLoss ignores them
            labels[labels == self.tokenizer.pad_token_id] = -100

            item["labels"] = labels
            item["answer_text"] = target_text

        return item


def _load_and_cache_data(file_path, cache_name, load_cached_data):
    """
    Helper function to load data from CSV or Parquet cache.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    # 2. Load from source
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Ensure text columns are strings
    text_cols = ["context", "question"]
    if "answer_text" in df.columns:
        text_cols.append("answer_text")

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def get_dataloaders(load_cached_data=True):
    """
    Initializes the tokenizer and creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load DataFrames with Caching Logic
    train_df = _load_and_cache_data(
        Config.TRAIN_PATH, "train_processed", load_cached_data
    )
    val_df = _load_and_cache_data(Config.VAL_PATH, "val_processed", load_cached_data)
    test_df = _load_and_cache_data(Config.TEST_PATH, "test_processed", load_cached_data)

    # Debugging: Subsample if configured
    if Config.DEBUG:
        train_df = train_df.head(20)
        val_df = val_df.head(10)
        test_df = test_df.head(10)

    # Create Datasets
    train_dataset = QADataset(
        train_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        is_test=False,
    )

    val_dataset = QADataset(
        val_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        is_test=False,
    )

    test_dataset = QADataset(
        test_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        is_test=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, tokenizer
