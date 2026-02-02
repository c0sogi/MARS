import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles tokenization of text data.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["Comment"].values
        if not self.is_test:
            self.targets = df[Config.target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.texts[index])

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

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            target = torch.tensor(self.targets[index], dtype=torch.float)
            item["target"] = target

        return item


def load_process_cache_data(source_path, cache_path, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes source CSV and caches it.
    Processing involves decoding the text column.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Re-processing.")

    # Process from scratch
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Decode text
    if "Comment" in df.columns:
        df["Comment"] = df["Comment"].apply(decode_text)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    load_cached_data=True,
    debug=Config.debug,
    debug_sample_size=Config.debug_sample_size,
):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    """

    # Define cache paths
    train_cache = os.path.join(Config.output_dir, "train_decoded.parquet")
    val_cache = os.path.join(Config.output_dir, "val_decoded.parquet")
    test_cache = os.path.join(Config.output_dir, "test_decoded.parquet")

    # Load data
    train_df = load_process_cache_data(Config.train_path, train_cache, load_cached_data)
    val_df = load_process_cache_data(Config.val_path, val_cache, load_cached_data)
    test_df = load_process_cache_data(Config.test_path, test_cache, load_cached_data)

    # Handle Debug Mode
    if debug:
        train_df = train_df.head(debug_sample_size).reset_index(drop=True)
        val_df = val_df.head(debug_sample_size).reset_index(drop=True)
        test_df = test_df.head(debug_sample_size).reset_index(drop=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_path)

    # Create Datasets
    train_dataset = InsultDataset(train_df, tokenizer, Config.max_len, is_test=False)
    val_dataset = InsultDataset(val_df, tokenizer, Config.max_len, is_test=False)
    test_dataset = InsultDataset(test_df, tokenizer, Config.max_len, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
