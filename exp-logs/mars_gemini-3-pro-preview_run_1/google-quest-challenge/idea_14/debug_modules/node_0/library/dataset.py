import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class StackExchangeDataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Prepares dual-stream inputs:
    1. Question Stream: Title + Body
    2. Answer Stream: Title + Answer (Contextualized)
    """

    def __init__(self, data, tokenizer, target_cols=None, max_len=512, is_test=False):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the data.
            tokenizer: Transformers tokenizer.
            target_cols (list): List of target column names.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract text columns to lists for faster access
        self.titles = self.data["question_title"].fillna("").astype(str).tolist()
        self.bodies = self.data["question_body"].fillna("").astype(str).tolist()
        self.answers = self.data["answer"].fillna("").astype(str).tolist()
        self.qa_ids = self.data["qa_id"].tolist()

        # Pre-extract labels if training/validation
        if not self.is_test:
            if self.target_cols is None:
                raise ValueError(
                    "target_cols must be provided for training/validation sets."
                )
            self.labels = self.data[self.target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # --- Stream A: Question (Title + Body) ---
        # We prioritize the Title (first seq), so we truncate the Body (second seq) if needed.
        inputs_q = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
        )

        # --- Stream B: Contextualized Answer (Title + Answer) ---
        # We inject Title to provide context for the Answer encoder.
        # We prioritize the Title, so we truncate the Answer if needed.
        inputs_a = self.tokenizer(
            title,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
        )

        # --- Pooling Mask for Answer ---
        # We want to pool only the 'Answer' tokens from Stream B.
        # sequence_ids() returns:
        #   None: Special tokens
        #   0: First sequence (Title)
        #   1: Second sequence (Answer)
        seq_ids = inputs_a.sequence_ids()
        # Create mask: 1.0 for Answer tokens, 0.0 for Title and Special tokens
        pooling_mask_a = [1.0 if s == 1 else 0.0 for s in seq_ids]

        # Construct return dictionary
        item = {
            "input_ids_q": torch.tensor(inputs_q["input_ids"], dtype=torch.long),
            "attention_mask_q": torch.tensor(
                inputs_q["attention_mask"], dtype=torch.long
            ),
            "input_ids_a": torch.tensor(inputs_a["input_ids"], dtype=torch.long),
            "attention_mask_a": torch.tensor(
                inputs_a["attention_mask"], dtype=torch.long
            ),
            "pooling_mask_a": torch.tensor(pooling_mask_a, dtype=torch.float),
            "qa_id": torch.tensor(self.qa_ids[idx], dtype=torch.long),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_target_columns():
    """
    Retrieves the list of 30 target column names from the sample submission file.
    """
    if not os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {Config.SAMPLE_SUBMISSION_PATH}"
        )

    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    return [col for col in sample_sub.columns if col != "qa_id"]


def load_data(split="train", load_cached_data=True):
    """
    Loads the dataset for the specified split.
    Implements caching using Parquet files to optimize loading times.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The loaded data.
    """
    # Map split to paths
    if split == "train":
        meta_path = Config.TRAIN_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif split == "val":
        meta_path = Config.VAL_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif split == "test":
        meta_path = Config.TEST_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading {split} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Error loading cache for {split}: {e}. Falling back to metadata.")

    # 2. Load from metadata
    # print(f"Loading {split} data from metadata: {meta_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # 3. Basic Preprocessing
    # Ensure text columns are strings and handle NaNs
    text_cols = ["question_title", "question_body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 4. Save to cache
    try:
        # print(f"Saving {split} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    return df
