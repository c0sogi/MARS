import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


def load_and_cache_data(csv_path, cache_name, load_cached_data=True):
    """
    Loads data from CSV, caches it as Parquet to the working directory,
    or loads from Parquet if available to satisfy deterministic processing requirements.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name to use for the cached file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # Fallback to processing from scratch if cache load fails
            pass

    # 2. Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not cache data to {cache_path}: {e}")

    return df


def get_tokenizer(model_name=None):
    """
    Helper to load the tokenizer defined in Config.
    """
    if model_name is None:
        model_name = Config.model_name
    return AutoTokenizer.from_pretrained(model_name)


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.max_len, mode="train"):
        """
        PyTorch Dataset for StackExchange Question-Answer pairs.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            tokenizer (PreTrainedTokenizer): The transformer tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Pre-extract text columns to numpy arrays for faster access in __getitem__
        # Fill NaNs with empty strings to prevent tokenization errors
        self.titles = df["question_title"].fillna("").astype(str).values
        self.bodies = df["question_body"].fillna("").astype(str).values
        self.answers = df["answer"].fillna("").astype(str).values

        # Pre-extract targets if not in test mode
        if self.mode in ["train", "val"]:
            # Ensure we only get the specific target columns defined in Config
            self.targets = df[Config.target_cols].values.astype("float32")
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve raw text
        question_title = self.titles[idx]
        question_body = self.bodies[idx]
        answer = self.answers[idx]

        # Process Question Stream: [CLS] Title [SEP] Body [SEP]
        # Passing two arguments to tokenizer automatically adds separator tokens
        # which helps the model distinguish between intent (title) and context (body).
        q_inputs = self.tokenizer(
            question_title,
            question_body,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Process Answer Stream: [CLS] Answer [SEP]
        a_inputs = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Construct output dictionary
        # .squeeze(0) is required because return_tensors='pt' adds a batch dimension of 1
        item = {
            "q_input_ids": q_inputs["input_ids"].squeeze(0),
            "q_attention_mask": q_inputs["attention_mask"].squeeze(0),
            "a_input_ids": a_inputs["input_ids"].squeeze(0),
            "a_attention_mask": a_inputs["attention_mask"].squeeze(0),
        }

        # Add targets if available
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item
