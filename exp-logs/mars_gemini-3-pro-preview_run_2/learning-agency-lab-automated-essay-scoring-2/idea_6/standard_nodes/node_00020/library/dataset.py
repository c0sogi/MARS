import os
import hashlib
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Handles input_ids, attention_masks, and regression targets.
    """

    def __init__(self, encodings, scores=None):
        self.encodings = encodings
        self.scores = scores

    def __getitem__(self, idx):
        # Create item dictionary with tensors
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.scores is not None:
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def _get_data_hash(df: pd.DataFrame) -> str:
    """
    Generates a unique hash for the dataframe based on essay_ids.
    Used to create unique cache filenames for different folds/subsets.
    """
    # Concatenate all essay IDs to form a unique signature for this dataset subset
    # Using essay_id ensures that if the split changes (e.g. CV folds), the hash changes.
    ids_signature = "".join(df["essay_id"].astype(str).tolist())
    return hashlib.md5(ids_signature.encode("utf-8")).hexdigest()


def _process_and_cache(df: pd.DataFrame, tokenizer, load_cached_data: bool = True):
    """
    Tokenizes dataframe content with caching mechanism.

    Args:
        df: Input dataframe containing 'full_text' and 'essay_id'.
        tokenizer: Transformers tokenizer.
        load_cached_data: Whether to try loading from cache.

    Returns:
        dict: Dictionary containing 'input_ids' and 'attention_mask' tensors.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Generate unique filename based on data content
    data_hash = _get_data_hash(df)
    cache_path = os.path.join(Config.CACHE_DIR, f"tokens_{data_hash}.pt")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return torch.load(cache_path)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    texts = df["full_text"].fillna("").astype(str).tolist()

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # 3. Save to cache
    try:
        torch.save(encodings, cache_path)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return encodings


def get_dataloaders(train_df, val_df, tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        tokenizer: Initialized tokenizer.
        load_cached_data: Whether to use cached tokenization.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Process Train
    train_encodings = _process_and_cache(train_df, tokenizer, load_cached_data)
    train_scores = train_df["score"].values
    train_dataset = EssayDataset(train_encodings, train_scores)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize training
    )

    # Process Val
    val_encodings = _process_and_cache(val_df, tokenizer, load_cached_data)
    val_scores = val_df["score"].values
    val_dataset = EssayDataset(val_encodings, val_scores)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(test_df, tokenizer, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    encodings = _process_and_cache(test_df, tokenizer, load_cached_data)
    # Test set has no scores
    dataset = EssayDataset(encodings, scores=None)

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
