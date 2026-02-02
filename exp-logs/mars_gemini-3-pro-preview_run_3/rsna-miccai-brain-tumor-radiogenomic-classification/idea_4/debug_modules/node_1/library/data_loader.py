import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.utils import load_dataset


class MGMTDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Multi-Stream 2.5D Network.

    Expects input X of shape (N, 4, 32, 256, 256).
    Returns 4 separate tensors for the modalities and the target/ID.
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve sample: (4, 32, 256, 256)
        sample = self.X[idx]

        # Split into separate streams for the Siamese network
        # Each stream gets shape (32, 256, 256)
        flair = torch.from_numpy(sample[0])
        t1w = torch.from_numpy(sample[1])
        t1wce = torch.from_numpy(sample[2])
        t2w = torch.from_numpy(sample[3])

        if self.y is not None:
            # Training/Validation mode: return label
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return flair, t1w, t1wce, t2w, label
        else:
            # Test/Inference mode: return ID
            return flair, t1w, t1wce, t2w, self.ids[idx]


def get_dataloaders(
    batch_size=8, num_workers=2, load_cached_data=True, debug=False, input_dir="./input"
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for training/inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached .npy files.
        debug (bool): If True, limits dataset size for quick debugging.
        input_dir (str): Root directory of input data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure cache directory exists
    cache_dir = "./working/idea_4/"
    os.makedirs(cache_dir, exist_ok=True)

    # Metadata paths
    train_meta = "./metadata/train.parquet"
    val_meta = "./metadata/val.parquet"
    test_meta = "./metadata/test.parquet"

    # --- Load Training Data ---
    train_X, train_y, train_ids = load_dataset(
        train_meta, cache_dir, load_cached=load_cached_data, input_dir=input_dir
    )

    # --- Load Validation Data ---
    val_X, val_y, val_ids = load_dataset(
        val_meta, cache_dir, load_cached=load_cached_data, input_dir=input_dir
    )

    # --- Load Test Data ---
    test_X, test_y, test_ids = load_dataset(
        test_meta, cache_dir, load_cached=load_cached_data, input_dir=input_dir
    )

    # --- Debugging: Subset data ---
    if debug:
        print("Debug mode enabled: reducing dataset sizes.")
        limit = 16  # Small number for quick testing
        train_X, train_y, train_ids = (
            train_X[:limit],
            train_y[:limit],
            train_ids[:limit],
        )
        val_X, val_y, val_ids = val_X[:limit], val_y[:limit], val_ids[:limit]
        test_X, test_ids = test_X[:limit], test_ids[:limit]
        # test_y is None

    # --- Create Datasets ---
    train_dataset = MGMTDataset(train_X, train_y, train_ids)
    val_dataset = MGMTDataset(val_X, val_y, val_ids)
    test_dataset = MGMTDataset(test_X, test_y, test_ids)

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
