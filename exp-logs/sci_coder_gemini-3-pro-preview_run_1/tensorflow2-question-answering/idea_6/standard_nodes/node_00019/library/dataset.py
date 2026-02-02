import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_prep import process_data


class NQDataset(Dataset):
    """
    PyTorch Dataset for Natural Questions.
    Wraps the processed DataFrame containing tokenized questions and candidates.
    """

    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing processed examples.
                                    Expected columns: q_indices, c_indices, label_long,
                                    short_start, short_end, example_id, global_start, global_end.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert lists to numpy arrays first for efficiency, then to tensors
        # Data is stored as lists in parquet
        q_indices = np.array(row["q_indices"], dtype=np.int64)
        c_indices = np.array(row["c_indices"], dtype=np.int64)

        # Labels
        # Long answer is binary (0/1), use float for BCEWithLogitsLoss
        label_long = np.float32(row["label_long"])

        # Short answer indices are integers for CrossEntropyLoss
        short_start = np.int64(row["short_start"])
        short_end = np.int64(row["short_end"])

        return {
            "q_indices": torch.from_numpy(q_indices),
            "c_indices": torch.from_numpy(c_indices),
            "label_long": torch.tensor(label_long),
            "short_start": torch.tensor(short_start),
            "short_end": torch.tensor(short_end),
            "example_id": row["example_id"],
            "global_start": row["global_start"],
            "global_end": row["global_end"],
        }


def collate_fn(batch):
    """
    Custom collate function to batch dictionaries of data.
    Since sequences are already padded to fixed lengths in data_prep,
    we just stack them.
    """
    # Stack tensors
    q_indices = torch.stack([item["q_indices"] for item in batch])
    c_indices = torch.stack([item["c_indices"] for item in batch])
    label_long = torch.stack([item["label_long"] for item in batch])
    short_start = torch.stack([item["short_start"] for item in batch])
    short_end = torch.stack([item["short_end"] for item in batch])

    # Collect metadata as lists
    example_ids = [item["example_id"] for item in batch]
    global_starts = [item["global_start"] for item in batch]
    global_ends = [item["global_end"] for item in batch]

    return {
        "q_indices": q_indices,
        "c_indices": c_indices,
        "label_long": label_long,
        "short_start": short_start,
        "short_end": short_end,
        "example_ids": example_ids,
        "global_starts": global_starts,
        "global_ends": global_ends,
    }


def get_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Orchestrates the loading of data and creation of DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
                                 If False or load fails, re-runs processing.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader, word2idx)
    """
    # Load processed DataFrames using the provided data_prep module
    # This handles the caching logic and negative sampling internally
    train_df, val_df, test_df, word2idx = process_data(
        load_cached_data=load_cached_data
    )

    # Initialize Datasets
    train_dataset = NQDataset(train_df)
    val_dataset = NQDataset(val_df)
    test_dataset = NQDataset(test_df)

    # Initialize DataLoaders
    # Shuffle training data, keep val/test sequential
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, word2idx
