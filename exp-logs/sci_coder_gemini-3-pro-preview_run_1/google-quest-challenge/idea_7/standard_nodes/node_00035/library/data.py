import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    Dataset class for Question-Answer pairs.
    Handles dual-stream input (Question and Answer) and dual-target extraction.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Ensure text columns exist and handle NaNs
        # We assume 'question_text' has been created in preprocessing
        self.question_texts = df["question_text"].fillna("").astype(str).tolist()
        self.answer_texts = df["answer"].fillna("").astype(str).tolist()

        if not self.is_test:
            self.targets = df[Config.TARGET_COLS].values
            self.aux_targets = df[Config.QUESTION_TARGET_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.question_texts[idx]
        answer = self.answer_texts[idx]

        # Tokenize Question (Title + Body)
        q_encoded = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,  # Dynamic padding in Collate
            return_attention_mask=True,
        )

        # Tokenize Answer
        a_encoded = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        item = {
            "q_input_ids": q_encoded["input_ids"],
            "q_attention_mask": q_encoded["attention_mask"],
            "a_input_ids": a_encoded["input_ids"],
            "a_attention_mask": a_encoded["attention_mask"],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
            item["aux_labels"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)

        return item


class Collate:
    """
    Collate function for dynamic padding.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        q_input_ids = [item["q_input_ids"] for item in batch]
        q_attention_mask = [item["q_attention_mask"] for item in batch]
        a_input_ids = [item["a_input_ids"] for item in batch]
        a_attention_mask = [item["a_attention_mask"] for item in batch]

        # Pad Question inputs
        q_batch = self.tokenizer.pad(
            {"input_ids": q_input_ids, "attention_mask": q_attention_mask},
            padding=True,
            return_tensors="pt",
        )

        # Pad Answer inputs
        a_batch = self.tokenizer.pad(
            {"input_ids": a_input_ids, "attention_mask": a_attention_mask},
            padding=True,
            return_tensors="pt",
        )

        output = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
        }

        if "labels" in batch[0]:
            output["labels"] = torch.stack([item["labels"] for item in batch])
            output["aux_labels"] = torch.stack([item["aux_labels"] for item in batch])

        return output


def preprocess_df(df):
    """
    Combines question title and body into a single text column.
    """
    # Concatenate title and body with a space
    df["question_text"] = (
        df["question_title"].fillna("") + " " + df["question_body"].fillna("")
    )
    return df


def get_data(load_cached_data=True):
    """
    Loads data with caching mechanism.
    """
    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_cached.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cached.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cached.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from parquet...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Loading data from metadata CSVs and processing...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Preprocess
        train_df = preprocess_df(train_df)
        val_df = preprocess_df(val_df)
        test_df = preprocess_df(test_df)

        # Save to cache
        print(f"Saving cached data to {Config.WORKING_DIR}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load data
    train_df, val_df, test_df = get_data(load_cached_data=load_cached_data)

    if debug:
        print("Debug mode: using subset of data.")
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_ds = QADataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_ds = QADataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_ds = QADataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create Collate function
    collate_fn = Collate(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
