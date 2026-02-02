import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    """

    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def _load_raw_data(metadata_path, raw_path):
    """
    Merges metadata with raw text data based on ID.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    meta_df = pd.read_csv(metadata_path)
    raw_df = pd.read_csv(raw_path)

    # Merge to get text content
    # metadata contains IDs and labels, raw contains IDs and text
    df = pd.merge(meta_df, raw_df[["id", "comment_text"]], on="id", how="left")

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    return df


def _process_split(
    split_name, metadata_path, raw_path, tokenizer, config, load_cached_data
):
    """
    Handles loading, tokenizing, and caching for a specific data split.
    """
    # Define cache paths
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{split_name}_input_ids.npy")
    mask_path = os.path.join(cache_dir, f"{split_name}_attention_mask.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")

    # Determine if we need labels (Test set doesn't have training labels)
    is_test = split_name == "test"

    # Check if cache exists
    cache_exists = os.path.exists(ids_path) and os.path.exists(mask_path)
    if not is_test:
        cache_exists = cache_exists and os.path.exists(labels_path)

    # 1. IF load_cached_data is True: Try to load
    if load_cached_data and cache_exists:
        print(f"Loading cached data for {split_name} from {cache_dir}...")
        try:
            input_ids = np.load(ids_path)
            attention_mask = np.load(mask_path)
            labels = None
            if not is_test:
                labels = np.load(labels_path)
            return input_ids, attention_mask, labels
        except Exception as e:
            print(f"Failed to load cache for {split_name}: {e}. Recomputing...")
            # Fall through to recompute

    # 2. IF loading fails OR load_cached_data is False: Compute
    print(f"Processing and tokenizing data for {split_name}...")
    df = _load_raw_data(metadata_path, raw_path)

    # Handle Debug Mode
    if config.debug:
        print(f"DEBUG MODE: Truncating {split_name} to 1000 samples.")
        df = df.iloc[:1000].reset_index(drop=True)

    texts = df["comment_text"].tolist()

    # Tokenize
    # DeBERTa-v3 tokenizer handles special tokens automatically
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.max_len,
        return_tensors="np",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Save to cache
    np.save(ids_path, input_ids)
    np.save(mask_path, attention_mask)

    labels = None
    if not is_test:
        labels = df[config.labels].values.astype(np.float32)
        np.save(labels_path, labels)

    return input_ids, attention_mask, labels


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    config = Config()

    # Initialize Tokenizer
    # Using the model name from config (microsoft/deberta-v3-base)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # --- Train Split ---
    train_ids, train_mask, train_labels = _process_split(
        split_name="train",
        metadata_path=config.train_metadata_path,
        raw_path=config.train_raw_path,
        tokenizer=tokenizer,
        config=config,
        load_cached_data=load_cached_data,
    )

    # --- Val Split ---
    # Note: Val metadata comes from train.csv source
    val_ids, val_mask, val_labels = _process_split(
        split_name="val",
        metadata_path=config.val_metadata_path,
        raw_path=config.train_raw_path,
        tokenizer=tokenizer,
        config=config,
        load_cached_data=load_cached_data,
    )

    # --- Test Split ---
    test_ids, test_mask, _ = _process_split(
        split_name="test",
        metadata_path=config.test_metadata_path,
        raw_path=config.test_raw_path,
        tokenizer=tokenizer,
        config=config,
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = ToxicityDataset(train_ids, train_mask, train_labels)
    val_dataset = ToxicityDataset(val_ids, val_mask, val_labels)
    test_dataset = ToxicityDataset(test_ids, test_mask, labels=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
