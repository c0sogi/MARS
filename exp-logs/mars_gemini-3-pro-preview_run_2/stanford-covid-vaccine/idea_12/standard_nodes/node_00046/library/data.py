import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import pre-defined classes and functions from the library to avoid re-implementation
from library.config import (
    Config,
    RNADataset,
    load_or_process_data,
    get_structure_indices,
)
from library.utils import set_seed


def get_partner_indices(structure):
    """
    Parses the dot-bracket structure to generate a partner_index_map.
    Wrapper around the library function to satisfy module requirements.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.array: Array where arr[i] is the index of the partner of base i,
                  or -1 if unpaired.
    """
    return get_structure_indices(structure)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_size=None,
):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to try loading from cache first.
        debug_size (int, optional): If set, truncates the dataset to this size for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Load training data
    # The library function handles the caching logic:
    # 1. Check if cache exists & load_cached_data is True -> Load
    # 2. Else -> Process from metadata -> Save to cache -> Return
    X_train, idx_train, y_train, mask_train, _ = load_or_process_data(
        mode="train", load_cached_data=load_cached_data
    )

    # Load validation data
    X_val, idx_val, y_val, mask_val, _ = load_or_process_data(
        mode="val", load_cached_data=load_cached_data
    )

    # Optional debugging: slice datasets
    if debug_size is not None:
        X_train = X_train[:debug_size]
        idx_train = idx_train[:debug_size]
        y_train = y_train[:debug_size]
        mask_train = mask_train[:debug_size]

        X_val = X_val[:debug_size]
        idx_val = idx_val[:debug_size]
        y_val = y_val[:debug_size]
        mask_val = mask_val[:debug_size]

    # Instantiate Datasets
    train_dataset = RNADataset(X_train, idx_train, y_train, mask_train)
    val_dataset = RNADataset(X_val, idx_val, y_val, mask_val)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns the test DataLoader.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        tuple: (test_loader, test_ids)
    """
    # Load test data
    # Note: load_or_process_data for test returns zero-filled arrays for targets and mask
    X_test, idx_test, y_test, mask_test, ids = load_or_process_data(
        mode="test", load_cached_data=load_cached_data
    )

    test_dataset = RNADataset(X_test, idx_test, y_test, mask_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, ids
