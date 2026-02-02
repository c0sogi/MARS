import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Dict, Any, Optional

from library.config import Config


def load_and_preprocess_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and preprocesses data for a specific split (train, val, test).
    Implements caching using Parquet files.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {split} data from cache: {cache_path}")
            return df
        except Exception:
            # print(f"Failed to load cache for {split}, re-processing...")
            pass

    # 2. Compute from scratch
    if split == "train":
        input_path = Config.TRAIN_PATH
    elif split == "val":
        input_path = Config.VAL_PATH
    elif split == "test":
        input_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Fill NaNs in text columns to avoid errors during string concatenation
    text_cols = ["question_title", "question_body", "answer", "category", "host"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Textual Metadata Injection for Question Branch
    # Format: "Category: {category}. Host: {host}. {question_title} {question_body}"
    df["question_input"] = (
        "Category: "
        + df["category"]
        + ". Host: "
        + df["host"]
        + ". "
        + df["question_title"]
        + " "
        + df["question_body"]
    )

    # Answer Branch
    df["answer_input"] = df["answer"]

    # Ensure target columns exist (fill with 0 for test/inference if missing)
    # This ensures the Dataset class logic remains consistent
    if split == "test":
        for col in Config.TARGET_COLS:
            if col not in df.columns:
                df[col] = 0.0

    # Cast targets to float32 for PyTorch
    for col in Config.TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {split} data to cache: {cache_path}")
    except Exception as e:
        # print(f"Warning: Could not save cache to {cache_path}. Error: {e}")
        pass

    return df


class StackExchangeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = Config.MAX_LEN,
        is_test: bool = False,
    ):
        """
        Args:
            data (pd.DataFrame): Preprocessed dataframe.
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            is_test (bool): If True, returns dummy labels.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to lists/arrays for faster access in __getitem__
        self.question_inputs = self.data["question_input"].tolist()
        self.answer_inputs = self.data["answer_input"].tolist()
        self.qa_ids = self.data["qa_id"].tolist()

        if not self.is_test:
            self.labels = self.data[Config.TARGET_COLS].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question_text = self.question_inputs[index]
        answer_text = self.answer_inputs[index]

        # Tokenize Question
        # We don't pad here; we pad in the CollateFn for dynamic padding
        q_encoded = self.tokenizer(
            question_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,  # DistilRoBERTa doesn't use token_type_ids
        )

        # Tokenize Answer
        a_encoded = self.tokenizer(
            answer_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "q_input_ids": q_encoded["input_ids"],
            "q_attention_mask": q_encoded["attention_mask"],
            "a_input_ids": a_encoded["input_ids"],
            "a_attention_mask": a_encoded["attention_mask"],
            "qa_id": self.qa_ids[index],
        }

        if not self.is_test:
            item["labels"] = self.labels[index]
        else:
            # Dummy labels for test set to maintain consistent interface
            item["labels"] = [0.0] * Config.NUM_LABELS

        return item


class CollateFn:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Dynamically pads the batch to the longest sequence in the batch.
        """
        # Prepare lists for tokenizer.pad
        q_features = [
            {
                "input_ids": item["q_input_ids"],
                "attention_mask": item["q_attention_mask"],
            }
            for item in batch
        ]
        a_features = [
            {
                "input_ids": item["a_input_ids"],
                "attention_mask": item["a_attention_mask"],
            }
            for item in batch
        ]

        # Pad Question Branch
        q_batch = self.tokenizer.pad(q_features, padding=True, return_tensors="pt")

        # Pad Answer Branch
        a_batch = self.tokenizer.pad(a_features, padding=True, return_tensors="pt")

        # Stack labels and qa_ids
        # Convert list of arrays to single array first to avoid warning/overhead
        labels = torch.tensor(
            np.array([item["labels"] for item in batch]), dtype=torch.float32
        )
        qa_ids = torch.tensor([item["qa_id"] for item in batch], dtype=torch.long)

        return {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
            "labels": labels,
            "qa_ids": qa_ids,
        }
