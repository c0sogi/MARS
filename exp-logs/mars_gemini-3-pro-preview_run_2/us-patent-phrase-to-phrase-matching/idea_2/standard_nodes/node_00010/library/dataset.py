import os
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_cpc_mapping

# Suppress verbose output from transformers
transformers.logging.set_verbosity_error()


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for Phrase Similarity Task.
    Prepares inputs for a Cross-Encoder model with Context Enrichment.
    """

    def __init__(self, df, tokenizer, max_length=128, is_test=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Prepare text inputs
        # We utilize the tokenizer's pair handling to create the structure:
        # [CLS] Context Description [SEP] Anchor [SEP] Target [SEP]
        # Text A: Context Description
        # Text B: Anchor + [SEP] + Target

        self.contexts = df["context_desc"].astype(str).tolist()

        # Ensure the separator has spaces to prevent token merging (e.g., "word[SEP]" vs "word [SEP]")
        sep_token = f" {tokenizer.sep_token} "
        self.pairs = (
            df["anchor"].astype(str) + sep_token + df["target"].astype(str)
        ).tolist()

        if not is_test:
            # Map float scores to discrete class indices for 5-class classification
            # 0.00 -> 0
            # 0.25 -> 1
            # 0.50 -> 2
            # 0.75 -> 3
            # 1.00 -> 4
            # We use round() to handle potential floating point inaccuracies
            self.labels = (df["score"] * 4).round().astype(int).values

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context_text = self.contexts[idx]
        pair_text = self.pairs[idx]

        # Tokenize
        # Passing two text arguments allows the tokenizer to automatically add special tokens
        # and segment IDs (if applicable) correctly.
        inputs = self.tokenizer(
            context_text,
            pair_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def process_data(split, load_cached_data=True):
    """
    Loads raw metadata, enriches it with Context Descriptions, and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'context_desc' column.
    """
    cache_file = f"{split}_processed.parquet"
    cache_path = os.path.join(Config.working_dir, cache_file)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails (e.g. corrupt file), fall through to processing
            pass

    # 2. Process Data from Metadata
    meta_path = os.path.join(Config.metadata_dir, f"{split}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Map Context Codes (e.g., "A47") to Descriptions (e.g., "Furniture...")
    cpc_map = get_cpc_mapping()
    # Use map, default to original code if not found
    df["context_desc"] = df["context"].map(cpc_map).fillna(df["context"])

    # 3. Save to Cache
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    # We use the tokenizer corresponding to the model architecture
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load Processed Data
    train_df = process_data("train", load_cached_data=load_cached_data)
    val_df = process_data("val", load_cached_data=load_cached_data)
    test_df = process_data("test", load_cached_data=load_cached_data)

    # Debugging: Subsample if configured
    if Config.debug:
        train_df = train_df.iloc[: Config.debug_sample_size]
        val_df = val_df.iloc[: Config.debug_sample_size]
        test_df = test_df.iloc[: Config.debug_sample_size]

    # Instantiate Datasets
    train_dataset = PhraseDataset(
        train_df, tokenizer, max_length=Config.max_length, is_test=False
    )

    val_dataset = PhraseDataset(
        val_df, tokenizer, max_length=Config.max_length, is_test=False
    )

    test_dataset = PhraseDataset(
        test_df, tokenizer, max_length=Config.max_length, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
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
