import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from library.config import Config
from library.utils import seed_everything


class StackExchangeDataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Handles dual-encoder input formatting:
    - Question Branch: Title + Body
    - Answer Branch: Answer
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len_q: int = Config.MAX_LEN_Q,
        max_len_a: int = Config.MAX_LEN_A,
        is_test: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len_q = max_len_q
        self.max_len_a = max_len_a
        self.is_test = is_test

        # Convert to numpy arrays for faster access and memory efficiency
        self.titles = self.data["question_title"].values
        self.bodies = self.data["question_body"].values
        self.answers = self.data["answer"].values

        if not self.is_test:
            self.targets = self.data[Config.TARGET_COLS].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        title = self.titles[index]
        body = self.bodies[index]
        answer = self.answers[index]

        # Construct inputs
        # Question: Concatenate title and body
        q_text = f"{title} {body}"
        # Answer: Just the answer text
        a_text = f"{answer}"

        # Tokenize Question (No padding here, handled in Collate)
        q_encoded = self.tokenizer(
            q_text,
            add_special_tokens=True,
            max_length=self.max_len_q,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # Tokenize Answer
        a_encoded = self.tokenizer(
            a_text,
            add_special_tokens=True,
            max_length=self.max_len_a,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        sample = {
            "q_input_ids": q_encoded["input_ids"],
            "a_input_ids": a_encoded["input_ids"],
        }

        if not self.is_test:
            sample["labels"] = torch.tensor(self.targets[index], dtype=torch.float)

        return sample


class Collate:
    """
    Collate function for dynamic padding.
    Pads sequences to the longest length in the batch.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract lists of input_ids
        q_input_ids = [item["q_input_ids"] for item in batch]
        a_input_ids = [item["a_input_ids"] for item in batch]

        # Dynamic padding for Question inputs
        q_batch = self.tokenizer.pad(
            {"input_ids": q_input_ids}, padding=True, return_tensors="pt"
        )

        # Dynamic padding for Answer inputs
        a_batch = self.tokenizer.pad(
            {"input_ids": a_input_ids}, padding=True, return_tensors="pt"
        )

        output = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
        }

        # Stack labels if they exist
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            output["labels"] = labels

        return output


def process_data(
    input_path: str, cache_path: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads data from CSV or Parquet cache.
    Fills NaNs in text columns.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, fall back to processing
            pass

    # Process from scratch
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Fill NaNs in text columns to avoid tokenization errors
    text_cols = ["question_title", "question_body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load and process data
    train_df = process_data(Config.TRAIN_PATH, Config.TRAIN_CACHE, load_cached_data)
    val_df = process_data(Config.VAL_PATH, Config.VAL_CACHE, load_cached_data)
    test_df = process_data(Config.TEST_PATH, Config.TEST_CACHE, load_cached_data)

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.iloc[:100].reset_index(drop=True)
        val_df = val_df.iloc[:100].reset_index(drop=True)
        test_df = test_df.iloc[:100].reset_index(drop=True)

    # Initialize Datasets
    train_dataset = StackExchangeDataset(
        train_df,
        tokenizer,
        max_len_q=Config.MAX_LEN_Q,
        max_len_a=Config.MAX_LEN_A,
        is_test=False,
    )
    val_dataset = StackExchangeDataset(
        val_df,
        tokenizer,
        max_len_q=Config.MAX_LEN_Q,
        max_len_a=Config.MAX_LEN_A,
        is_test=False,
    )
    test_dataset = StackExchangeDataset(
        test_df,
        tokenizer,
        max_len_q=Config.MAX_LEN_Q,
        max_len_a=Config.MAX_LEN_A,
        is_test=True,
    )

    # Initialize Collate
    collate_fn = Collate(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
