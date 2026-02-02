import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


def clean_text(text):
    """
    Applies minimal text cleaning (whitespace normalization).
    """
    if not isinstance(text, str):
        return ""
    # Replace multiple whitespaces with single space and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split, applying preprocessing and caching.

    Args:
        split (str): One of "train", "val", "test".
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_filename = f"{split}_processed.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return _handle_debug(df)
        except Exception:
            # If loading fails (e.g. corrupt file), proceed to recompute
            pass

    # 2. Compute/Process from scratch
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)

    # Apply preprocessing
    if "full_text" in df.columns:
        df["full_text"] = df["full_text"].apply(clean_text)

    # Save to cache (save full dataset before debug slicing)
    # Ensure directory exists again just in case
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return _handle_debug(df)


def _handle_debug(df):
    """
    Slices the dataframe if DEBUG mode is enabled in Config.
    """
    if Config.DEBUG:
        return df.head(Config.DEBUG_SAMPLE_SIZE)
    return df


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'full_text' and optionally 'score'.
            tokenizer: Transformers tokenizer instance.
            max_length (int, optional): Maximum sequence length. Defaults to Config.MAX_LENGTH.
        """
        self.texts = df["full_text"].astype(str).tolist()
        self.scores = df["score"].tolist() if "score" in df.columns else None
        self.essay_ids = df["essay_id"].tolist() if "essay_id" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else Config.MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by return_tensors="pt"
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Include essay_id if available (useful for inference/debugging)
        if self.essay_ids is not None:
            item["essay_id"] = self.essay_ids[idx]

        # Include score if available (for training/validation)
        if self.scores is not None:
            # Regression target: float
            # Config uses SmoothL1Loss, which expects float inputs
            score = torch.tensor(self.scores[idx], dtype=torch.float)
            item["score"] = score

        return item
