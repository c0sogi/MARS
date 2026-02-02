import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class JigsawDataset(Dataset):
    """
    PyTorch Dataset for the Jigsaw Toxicity Classification task.
    Loads pre-tokenized inputs and targets from numpy arrays.
    """

    def __init__(self, input_ids, attention_masks, targets=None, identities=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.identities = identities

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.identities is not None:
            item["identities"] = torch.tensor(self.identities[idx], dtype=torch.float)

        return item


def tokenize_and_cache(
    mode, metadata_path, text_path, tokenizer, max_len, load_cached_data=True
):
    """
    Loads data, tokenizes text, and caches the result as .npy files.

    Args:
        mode (str): 'train', 'val', or 'test'. Used for file naming.
        metadata_path (str): Path to the metadata CSV.
        text_path (str): Path to the raw text CSV.
        tokenizer: Transformers tokenizer instance.
        max_len (int): Maximum sequence length.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (input_ids, attention_masks, targets, identities)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")
    masks_path = os.path.join(cache_dir, f"{mode}_masks.npy")
    targets_path = os.path.join(cache_dir, f"{mode}_targets.npy")
    identities_path = os.path.join(cache_dir, f"{mode}_identities.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(ids_path) and os.path.exists(masks_path):
            # Check if targets exist (only for train/val)
            has_targets = os.path.exists(targets_path)

            print(f"Loading {mode} data from cache...")
            input_ids = np.load(ids_path)
            attention_masks = np.load(masks_path)

            targets = np.load(targets_path) if has_targets else None
            identities = (
                np.load(identities_path) if os.path.exists(identities_path) else None
            )

            return input_ids, attention_masks, targets, identities

    print(f"Processing {mode} data from scratch...")

    # 1. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    # Debugging: Subsample if configured
    if Config.DEBUG:
        meta_df = meta_df.iloc[:5000].copy()
        print(f"DEBUG mode: Subsampled {mode} to {len(meta_df)} rows.")

    # 2. Load Raw Text
    # We only need 'id' and 'comment_text'
    text_df = pd.read_csv(text_path, usecols=["id", "comment_text"])

    # 3. Merge
    # Inner join ensures we only get text for the IDs in our specific split (train/val/test)
    df = meta_df.merge(text_df, on="id", how="inner")

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("missing")

    # 4. Tokenize
    # This returns a dictionary with 'input_ids' and 'attention_mask'
    encoded = tokenizer.batch_encode_plus(
        df["comment_text"].tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",  # Return numpy arrays directly
    )

    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]

    # 5. Extract Targets
    targets = None
    identities = None

    if Config.TARGET_COL in df.columns:
        targets = df[Config.TARGET_COL].values.astype(np.float32)

        # Extract identity columns if they exist
        id_cols = [c for c in Config.IDENTITY_COLS if c in df.columns]
        if id_cols:
            # Fill NaNs in identity columns with 0 (assumption: not mentioned)
            identities = df[id_cols].fillna(0.0).values.astype(np.float32)

    # 6. Save to Cache
    np.save(ids_path, input_ids)
    np.save(masks_path, attention_masks)

    if targets is not None:
        np.save(targets_path, targets)

    if identities is not None:
        np.save(identities_path, identities)

    return input_ids, attention_masks, targets, identities


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files if available.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    # We use the tokenizer corresponding to the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Save tokenizer for inference usage
    tokenizer.save_pretrained(Config.TOKENIZER_SAVE_DIR)

    # --- Process Training Data ---
    train_ids, train_masks, train_y, train_aux = tokenize_and_cache(
        mode="train",
        metadata_path=Config.TRAIN_METADATA_PATH,
        text_path=Config.TRAIN_TEXT_PATH,
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN,
        load_cached_data=load_cached_data,
    )

    train_dataset = JigsawDataset(train_ids, train_masks, train_y, train_aux)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Process Validation Data ---
    val_ids, val_masks, val_y, val_aux = tokenize_and_cache(
        mode="val",
        metadata_path=Config.VALIDATION_METADATA_PATH,
        text_path=Config.TRAIN_TEXT_PATH,  # Validation is a split of the original train file
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN,
        load_cached_data=load_cached_data,
    )

    val_dataset = JigsawDataset(val_ids, val_masks, val_y, val_aux)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Process Test Data ---
    test_ids, test_masks, _, _ = tokenize_and_cache(
        mode="test",
        metadata_path=Config.TEST_METADATA_PATH,
        text_path=Config.TEST_TEXT_PATH,
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN,
        load_cached_data=load_cached_data,
    )

    # Test dataset has no targets
    test_dataset = JigsawDataset(test_ids, test_masks, targets=None, identities=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
