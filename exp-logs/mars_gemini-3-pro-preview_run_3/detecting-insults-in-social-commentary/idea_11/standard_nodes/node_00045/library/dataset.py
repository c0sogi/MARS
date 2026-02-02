import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization and prepares tensors for the model.
    Supports both hard labels (0/1) and soft targets (probabilities) for distillation.
    """

    def __init__(self, texts, tokenizer, max_len, targets=None):
        """
        Args:
            texts (list or pd.Series): List of text comments.
            tokenizer: HuggingFace tokenizer instance.
            max_len (int): Maximum sequence length for tokenization.
            targets (list or pd.Series, optional): Target labels (int or float). Defaults to None.
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,  # Not strictly necessary for RoBERTa/DeBERTa logic here
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.targets is not None:
            # Convert to float to support both binary labels and soft probabilities
            # BCEWithLogitsLoss expects float targets
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            output["target"] = target

        return output


def load_and_process_data(mode, config=None, load_cached_data=True, debug=False):
    """
    Loads dataset from metadata, applies preprocessing (decoding), and handles caching.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        config (Config, optional): Configuration object. Defaults to None.
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    cfg = config if config is not None else Config

    # Determine file paths based on mode
    if mode == "train":
        source_path = cfg.train_path
        cache_filename = "train_processed.parquet"
    elif mode == "val":
        source_path = cfg.val_path
        cache_filename = "val_processed.parquet"
    elif mode == "test":
        source_path = cfg.test_path
        cache_filename = "test_processed.parquet"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")

    cache_path = os.path.join(cfg.cache_dir, cache_filename)

    # Ensure cache directory exists
    os.makedirs(cfg.cache_dir, exist_ok=True)

    df = None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )
            df = None

    # If not loaded from cache, process from source
    if df is None:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        df = pd.read_csv(source_path)

        # Apply deterministic preprocessing
        if "Comment" in df.columns:
            df["Comment"] = df["Comment"].apply(decode_text)
        else:
            raise ValueError(f"Column 'Comment' missing in {source_path}")

        # Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

    # Handle Debugging (Subsampling)
    # We subsample AFTER loading/caching to ensure the cache always contains the full dataset
    if debug:
        df = df.iloc[: cfg.debug_sample_size].reset_index(drop=True)

    return df
