import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Tokenizes text content and provides input_ids, attention_mask, and targets.
    """

    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, is_test=False):
        # Ensure text is string and handle potential NaNs
        self.texts = df["Comment"].astype(str).values
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not self.is_test:
            self.targets = df["Insult"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        item = {"input_ids": ids, "attention_mask": mask}

        if not self.is_test:
            # Use float for BCEWithLogitsLoss
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def load_processed_data(
    file_path,
    cache_filename,
    load_cached_data=True,
    debug=False,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads data from CSV, applies deterministic processing (decoding), and caches to Parquet.

    Args:
        file_path (str): Path to the source CSV file.
        cache_filename (str): Name of the cache file (e.g., 'train.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, returns a small subset of the data.
        debug_sample_size (int): Number of samples to return in debug mode.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    df = None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # Fallback to processing if cache is corrupt
            df = None

    # 2. Process from scratch if needed
    if df is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Apply deterministic processing: Decode unicode text
        if "Comment" in df.columns:
            df["Comment"] = df["Comment"].apply(decode_text)

        # Save to cache for future runs
        df.to_parquet(cache_path, index=False)

    # 3. Apply debug sampling
    # We slice after loading/caching to ensure the cache always contains the full dataset
    if debug:
        df = df.iloc[:debug_sample_size].reset_index(drop=True)

    return df


def get_dataloader(
    df,
    tokenizer,
    batch_size,
    shuffle=False,
    max_len=Config.MAX_LEN,
    is_test=False,
    **kwargs,
):
    """
    Creates a PyTorch DataLoader for the given DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        tokenizer: Transformer tokenizer.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        max_len (int): Maximum sequence length.
        is_test (bool): Whether this is test data (no labels).
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = InsultDataset(df, tokenizer, max_len=max_len, is_test=is_test)

    loader_args = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.PIN_MEMORY,
        "drop_last": False,
    }

    # Allow overriding defaults via kwargs
    loader_args.update(kwargs)

    return DataLoader(dataset, **loader_args)
