import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from library.config import Config


def load_data(load_cached_data=True):
    """
    Loads train, validation, and test dataframes.
    Implements caching using Parquet to speed up subsequent loads.
    Handles debug mode slicing.
    """
    # Define cache paths
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cached.parquet")
    val_cache = os.path.join(cache_dir, "val_cached.parquet")
    test_cache = os.path.join(cache_dir, "test_cached.parquet")

    dfs = {}
    files = [
        ("train", Config.train_path, train_cache),
        ("val", Config.val_path, val_cache),
        ("test", Config.test_path, test_cache),
    ]

    for name, csv_path, cache_path in files:
        df = None
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                # If cache load fails, fall back to CSV
                pass

        # 2. If not loaded, read from CSV and cache
        if df is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} not found.")

            df = pd.read_csv(csv_path)

            # Ensure text columns are strings and fill NaNs
            text_cols = ["question_title", "question_body", "answer"]
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

            # Save to cache
            df.to_parquet(cache_path, index=False)

        # Handle Debug Mode
        if Config.debug:
            df = df.head(Config.debug_sample_size).reset_index(drop=True)

        dfs[name] = df

    return dfs["train"], dfs["val"], dfs["test"]


class QuestDataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Concatenates title and body for the question input.
    Uses answer for the answer input.
    Returns unpadded token lists for dynamic padding in Collate.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        is_test: bool = False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.target_cols = Config.target_cols

        # Pre-extract data to lists for faster access
        self.q_titles = df["question_title"].tolist()
        self.q_bodies = df["question_body"].tolist()
        self.answers = df["answer"].tolist()

        if not self.is_test:
            self.labels = df[self.target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Prepare text
        q_text = self.q_titles[idx] + " " + self.q_bodies[idx]
        a_text = self.answers[idx]

        # Tokenize Question
        # We do NOT pad here. We only truncate. Padding happens in Collate.
        q_encoded = self.tokenizer(
            q_text,
            add_special_tokens=True,
            max_length=Config.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Tokenize Answer
        a_encoded = self.tokenizer(
            a_text,
            add_special_tokens=True,
            max_length=Config.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "q_input_ids": q_encoded["input_ids"],
            "q_attention_mask": q_encoded["attention_mask"],
            "a_input_ids": a_encoded["input_ids"],
            "a_attention_mask": a_encoded["attention_mask"],
        }

        if not self.is_test:
            item["labels"] = self.labels[idx]

        return item


class Collate:
    """
    Collate function to perform dynamic padding.
    Pads sequences to the longest sequence in the batch, independent for Q and A.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of dicts from __getitem__

        # 1. Process Question Inputs
        q_input_ids = [item["q_input_ids"] for item in batch]
        q_attention_mask = [item["q_attention_mask"] for item in batch]

        # Pad Q inputs
        q_batch = self.tokenizer.pad(
            {"input_ids": q_input_ids, "attention_mask": q_attention_mask},
            padding="longest",
            return_tensors="pt",
        )

        # 2. Process Answer Inputs
        a_input_ids = [item["a_input_ids"] for item in batch]
        a_attention_mask = [item["a_attention_mask"] for item in batch]

        # Pad A inputs
        a_batch = self.tokenizer.pad(
            {"input_ids": a_input_ids, "attention_mask": a_attention_mask},
            padding="longest",
            return_tensors="pt",
        )

        # Construct final batch dictionary
        final_batch = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
        }

        # 3. Process Labels if present
        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            final_batch["labels"] = torch.tensor(np.array(labels), dtype=torch.float)

        return final_batch
