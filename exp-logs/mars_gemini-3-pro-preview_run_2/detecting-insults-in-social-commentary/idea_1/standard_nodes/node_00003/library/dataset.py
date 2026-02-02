import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.vocabulary import build_vocabulary, Vocabulary
from library.utils import set_seed

# Set seed for reproducibility
set_seed(SEED)


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Stores tokenized indices and labels.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): DataFrame containing 'indices' (list of ints) and 'Insult' (int/float).
        """
        self.indices = data["indices"].tolist()
        # Handle cases where 'Insult' might be missing (e.g., test set inference)
        if "Insult" in data.columns:
            self.labels = data["Insult"].fillna(0).astype(float).tolist()
        else:
            self.labels = [0.0] * len(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            indices (torch.Tensor): LongTensor of token indices.
            label (float): The label (0.0 or 1.0).
        """
        indices = torch.tensor(self.indices[idx], dtype=torch.long)
        label = self.labels[idx]
        return indices, label


def collate_fn(
    batch: List[Tuple[torch.Tensor, float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for BiLSTM (padded sequences).

    Args:
        batch: List of (indices, label) tuples.

    Returns:
        text (torch.Tensor): Padded tensor of token indices (batch_size, max_len).
        lengths (torch.Tensor): Tensor of sequence lengths.
        labels (torch.Tensor): FloatTensor of labels.
    """
    indices_list, labels_list = zip(*batch)

    # Calculate lengths
    lengths = torch.tensor([len(x) for x in indices_list], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> (batch, seq_len)
    text = torch.nn.utils.rnn.pad_sequence(
        indices_list, batch_first=True, padding_value=0
    )

    # Stack labels
    labels = torch.tensor(labels_list, dtype=torch.float)

    return text, lengths, labels


def process_dataframe(df: pd.DataFrame, vocab: Vocabulary) -> pd.DataFrame:
    """
    Tokenizes the 'Comment' column in the dataframe using the provided vocabulary.
    Returns a new dataframe with 'indices' and 'Insult' columns.
    """
    # Ensure Comment column is string
    comments = df["Comment"].astype(str).tolist()

    # Transform text to indices
    indices_list = [vocab.transform(text) for text in comments]

    # Create processed dataframe
    processed_df = pd.DataFrame({"indices": indices_list})

    if "Insult" in df.columns:
        processed_df["Insult"] = df["Insult"]

    return processed_df


def get_cached_dataframe(
    raw_path: str, vocab: Vocabulary, cache_name: str, load_cached_data: bool
) -> pd.DataFrame:
    """
    Loads raw data, processes it (tokenization), and handles caching.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_tokens.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure indices are lists (parquet might load as numpy arrays depending on engine)
            # but usually read_parquet handles list columns correctly with pyarrow.
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Re-processing.")

    # 2. Process from scratch
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    processed_df = process_dataframe(raw_df, vocab)

    # 3. Save to cache
    try:
        processed_df.to_parquet(cache_path, index=False)
        print(f"Saved processed data to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return processed_df


def get_dataloaders(
    batch_size: int = BATCH_SIZE, load_cached_data: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry point to get DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to use cached vocabulary and processed data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Raw Training Data to build Vocabulary
    # We need the raw text to build the vocab.
    # Even if we load cached processed data later, we need the vocab object for consistency check or inference.
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_PATH}")

    train_df_raw = pd.read_csv(TRAIN_PATH)
    train_texts = train_df_raw["Comment"].astype(str).tolist()

    # 2. Build or Load Vocabulary
    vocab = build_vocabulary(texts=train_texts, load_cached_data=load_cached_data)

    # 3. Get Processed DataFrames (with caching)
    train_df = get_cached_dataframe(TRAIN_PATH, vocab, "train", load_cached_data)
    val_df = get_cached_dataframe(VAL_PATH, vocab, "val", load_cached_data)
    test_df = get_cached_dataframe(TEST_PATH, vocab, "test", load_cached_data)

    # 4. Create Datasets
    train_dataset = InsultDataset(train_df)
    val_dataset = InsultDataset(val_df)
    test_dataset = InsultDataset(test_df)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
