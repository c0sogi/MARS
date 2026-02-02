import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification using DeBERTa-v3.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not self.is_test:
            self.labels = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.texts[index])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            item["labels"] = labels

        return item


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for a specific split.
    Handles merging metadata with raw text and caching the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe with text and labels.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.train_metadata_path
        cache_path = Config.train_cache_path
    elif split == "val":
        meta_path = Config.val_metadata_path
        cache_path = Config.val_cache_path
    elif split == "test":
        meta_path = Config.test_metadata_path
        cache_path = Config.test_cache_path
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Handle debug mode after loading cache
            if Config.debug:
                df = df.sample(
                    n=min(100, len(df)), random_state=Config.seed
                ).reset_index(drop=True)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source (Metadata + Raw)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Identify unique source files referenced in metadata
    sources = meta_df["source_file"].unique()
    loaded_data = []

    for src in sources:
        src_path = os.path.join(Config.input_dir, src)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Raw source file {src} missing at {src_path}")

        src_df = pd.read_csv(src_path)

        # Filter metadata for this source
        subset_meta = meta_df[meta_df["source_file"] == src]

        # Merge to get text content
        # We use inner join on ID to attach text from src_df to labels in subset_meta
        merged = pd.merge(
            subset_meta, src_df[["id", "comment_text"]], on="id", how="inner"
        )
        loaded_data.append(merged)

    df = pd.concat(loaded_data, ignore_index=True)

    # Fill missing text
    df["comment_text"] = df["comment_text"].fillna("")

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    # Handle debug mode
    if Config.debug:
        df = df.sample(n=min(100, len(df)), random_state=Config.seed).reset_index(
            drop=True
        )

    return df


def make_loader(df, tokenizer, is_train=True, batch_size=Config.train_batch_size):
    """
    Creates a DataLoader for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        tokenizer: Transformer tokenizer.
        is_train (bool): Whether this is for training (enables shuffle).
        batch_size (int): Batch size.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Determine if this is the test set based on columns
    # If target columns are missing, treat as test
    is_test = not all(col in df.columns for col in Config.target_cols)

    dataset = ToxicityDataset(
        df=df, tokenizer=tokenizer, max_len=Config.max_len, is_test=is_test
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )

    return loader
