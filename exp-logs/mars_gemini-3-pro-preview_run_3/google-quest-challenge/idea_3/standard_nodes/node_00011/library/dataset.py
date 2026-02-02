import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import PathConfig, TARGET_COLS


def load_data(load_cached_data=True):
    """
    Loads data from cache if available, otherwise loads from metadata,
    processes it (fills NaNs), caches it, and returns the DataFrames.
    """
    # Define cache paths
    train_cache = os.path.join(PathConfig.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(PathConfig.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(PathConfig.WORKING_DIR, "test_processed.parquet")

    # Ensure working directory exists
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)

    # Check if we should load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading data from cache: {PathConfig.WORKING_DIR}")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Loading data from metadata...")
    # Load from metadata
    train_df = pd.read_csv(PathConfig.TRAIN_META)
    val_df = pd.read_csv(PathConfig.VAL_META)
    test_df = pd.read_csv(PathConfig.TEST_META)

    # Basic Preprocessing (Fill NaNs)
    text_cols = ["question_title", "question_body", "answer"]
    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    # Save to cache
    print(f"Saving processed data to cache: {PathConfig.WORKING_DIR}")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing text and targets.
            tokenizer: Transformers tokenizer.
            max_len (int): Maximum sequence length.
            mode (str): 'train' (returns labels) or 'test' (no labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Pre-extract text data to lists for faster access
        self.titles = df["question_title"].values
        self.bodies = df["question_body"].values
        self.answers = df["answer"].values

        # Pre-extract targets if in training mode
        if self.mode == "train":
            # Ensure we only select the target columns
            self.targets = df[TARGET_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Construct Question and Answer segments
        question_text = title + " " + body
        answer_text = answer

        # Tokenize the pair
        # This will produce [CLS] Question [SEP] Answer [SEP] (or model equivalent)
        inputs = self.tokenizer(
            question_text,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate Segment Masks using sequence_ids()
        # sequence_ids() returns a list where:
        #   None -> Special tokens or Padding
        #   0    -> First sequence (Question)
        #   1    -> Second sequence (Answer)
        sequence_ids = inputs.sequence_ids()

        # Create masks for segment-aware pooling
        # We exclude special tokens (None) to get pure contextualized representations
        q_mask = [1 if s == 0 else 0 for s in sequence_ids]
        a_mask = [1 if s == 1 else 0 for s in sequence_ids]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        q_mask = torch.tensor(q_mask, dtype=torch.float)
        a_mask = torch.tensor(a_mask, dtype=torch.float)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "q_mask": q_mask,
            "a_mask": a_mask,
        }

        if self.mode == "train":
            labels = torch.tensor(self.targets[idx], dtype=torch.float)
            item["labels"] = labels

        return item
