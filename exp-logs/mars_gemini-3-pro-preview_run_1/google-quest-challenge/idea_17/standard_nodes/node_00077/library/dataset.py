import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


def load_data(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements a caching mechanism to save processed DataFrames to Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            try:
                print("Loading data from cache...")
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data.")

    # 2. Compute/Process data from scratch
    print("Loading data from metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Fill NaNs in text columns to ensure tokenizer stability
    text_cols = ["question_title", "question_body", "answer"]
    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    # 3. Save to cache
    print("Saving processed data to cache...")
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, val_df, test_df


class StackExchangeDataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Processes data into two separate streams:
    1. Question Stream: (Title, Body) pair
    2. Answer Stream: (Answer) only
    """

    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to numpy arrays for faster access
        self.titles = df["question_title"].values
        self.bodies = df["question_body"].values
        self.answers = df["answer"].values
        self.qa_ids = df["qa_id"].values

        if not self.is_test:
            self.targets = df[Config.TARGET_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Stream A: Question Context
        # Uses tokenizer's pair handling: [CLS] Title [SEP] Body [SEP]
        q_inputs = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # Dynamic padding handled in Collate
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # Stream B: Answer Context
        # [CLS] Answer [SEP]
        a_inputs = self.tokenizer(
            answer,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # Dynamic padding handled in Collate
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        item = {
            "qa_id": self.qa_ids[idx],
            "q_input_ids": q_inputs["input_ids"],
            "a_input_ids": a_inputs["input_ids"],
        }

        if not self.is_test:
            item["labels"] = self.targets[idx]

        return item


class Collate:
    """
    Collate function to handle dynamic padding for dual-stream inputs.
    Pads sequences to the longest length in the batch.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract sequences
        q_input_ids = [
            torch.tensor(item["q_input_ids"], dtype=torch.long) for item in batch
        ]
        a_input_ids = [
            torch.tensor(item["a_input_ids"], dtype=torch.long) for item in batch
        ]
        qa_ids = [item["qa_id"] for item in batch]

        # Dynamic Padding (Batch First)
        q_input_ids_padded = pad_sequence(
            q_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        a_input_ids_padded = pad_sequence(
            a_input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        # Create Attention Masks (1 for real tokens, 0 for padding)
        q_attention_mask = (q_input_ids_padded != self.pad_token_id).long()
        a_attention_mask = (a_input_ids_padded != self.pad_token_id).long()

        batch_out = {
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
            "q_input_ids": q_input_ids_padded,
            "q_attention_mask": q_attention_mask,
            "a_input_ids": a_input_ids_padded,
            "a_attention_mask": a_attention_mask,
        }

        # Stack targets if available
        if "labels" in batch[0]:
            labels = [torch.tensor(item["labels"], dtype=torch.float) for item in batch]
            batch_out["labels"] = torch.stack(labels)

        return batch_out
