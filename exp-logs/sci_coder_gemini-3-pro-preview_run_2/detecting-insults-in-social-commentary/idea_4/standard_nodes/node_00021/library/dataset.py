import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import clean_text


def get_tokenizer(model_name):
    """
    Loads the tokenizer for the specified model backbone.

    Args:
        model_name (str): The name of the model backbone (e.g., 'microsoft/deberta-v3-large').

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_dataset_dataframe(path, cache_name, load_cached_data=True):
    """
    Loads the dataset from CSV, cleans the text, and implements caching using parquet.
    Strictly follows the caching logic defined in the requirements.

    Args:
        path (str): Path to the input CSV file.
        cache_name (str): Name for the cached file (e.g., 'train_cleaned').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    cache_path = os.path.join(Config.working_dir, f"{cache_name}.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails (file missing or corrupt), proceed to compute from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False:
    # Compute/process the data from scratch.
    df = pd.read_csv(path)

    # Apply text cleaning using the imported utility
    if "Comment" in df.columns:
        df["Comment"] = df["Comment"].apply(clean_text)

    # Save the result to the cache directory
    df.to_parquet(cache_path, index=False)

    # 3. Return the data.
    return df


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization and target preparation.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Args:
            df (pd.DataFrame): The dataframe containing the data.
            tokenizer: The Hugging Face tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            is_test (bool): Whether this is a test dataset (no targets).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract values for faster access during training
        self.texts = df["Comment"].astype(str).values

        if not self.is_test:
            # Ensure target is float for BCEWithLogitsLoss
            # Assuming 'Insult' column exists for train/val sets
            self.targets = df["Insult"].values.astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten tensors to remove the batch dimension added by tokenizer (1, seq_len) -> (seq_len)
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            item["target"] = target

        return item
