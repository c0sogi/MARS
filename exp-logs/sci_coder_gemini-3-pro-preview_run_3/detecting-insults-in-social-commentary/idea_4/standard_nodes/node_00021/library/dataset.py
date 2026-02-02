import os
import codecs
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import ModelConfig


def decode_text(text):
    """
    Decodes unicode-escaped text.
    Handles cases where text might be NaN or not a string.
    """
    if pd.isna(text):
        return ""
    try:
        # Decode python string literal escape sequences (e.g. \n, \xe2)
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        # Fallback to original text if decoding fails
        return str(text)


def load_dataset_df(
    config: ModelConfig, split: str = "train", load_cached_data: bool = True
):
    """
    Loads the dataset for a specific split (train, val, test).
    Implements caching using Parquet files.

    Args:
        config: Configuration object containing paths.
        split: One of 'train', 'val', 'test'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Determine file paths based on split
    if split == "train":
        source_path = config.train_path
        cache_filename = "train_decoded.parquet"
    elif split == "val":
        source_path = config.val_path
        cache_filename = "val_decoded.parquet"
    elif split == "test":
        source_path = config.test_path
        cache_filename = "test_decoded.parquet"
    else:
        raise ValueError(f"Unknown split: {split}")

    cache_path = os.path.join(config.working_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Validate columns
    if "Comment" not in df.columns:
        raise ValueError(f"Column 'Comment' missing in {source_path}")

    # Apply deterministic processing
    df["Comment"] = df["Comment"].apply(decode_text)

    # 3. Save to cache
    os.makedirs(config.working_dir, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    """

    def __init__(self, texts, tokenizer, max_len, labels=None):
        """
        Args:
            texts (list or np.array): List of comment texts.
            tokenizer: Transformer tokenizer.
            max_len (int): Maximum sequence length.
            labels (list or np.array, optional): List of labels (0 or 1).
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten tensors (tokenizer returns [1, seq_len])
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.labels is not None:
            # Ensure label is a float for BCEWithLogitsLoss or Long for CrossEntropy
            # Usually for binary classification with one output node, float is needed.
            # If using 2 output nodes, Long is needed.
            # Assuming standard binary classification setup.
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            item["label"] = label

        return item
