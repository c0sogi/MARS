import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config, RNADataset, load_and_cache_data


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Maximum number of samples to load (for debugging).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Training Data
    # load_and_cache_data handles the caching logic internally:
    # 1. Checks if cache exists and load_cached_data is True.
    # 2. If not, processes from Parquet and saves to cache.
    train_data = load_and_cache_data(
        Config.TRAIN_PARQUET,
        "train_data",
        is_test=False,
        load_cached_data=load_cached_data,
    )
    train_ids, train_seq, train_loop, train_dist, train_tgt = train_data

    # Load Validation Data
    val_data = load_and_cache_data(
        Config.VAL_PARQUET, "val_data", is_test=False, load_cached_data=load_cached_data
    )
    val_ids, val_seq, val_loop, val_dist, val_tgt = val_data

    # Debugging: Slice datasets if max_samples is specified
    if max_samples is not None:
        train_seq = train_seq[:max_samples]
        train_loop = train_loop[:max_samples]
        train_dist = train_dist[:max_samples]
        train_tgt = train_tgt[:max_samples]

        val_seq = val_seq[:max_samples]
        val_loop = val_loop[:max_samples]
        val_dist = val_dist[:max_samples]
        val_tgt = val_tgt[:max_samples]

    # Create Datasets
    # RNADataset expects: sequences, loops, distances, targets
    train_ds = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    # Create DataLoaders
    # Pin memory is generally recommended for CUDA training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates and returns the test DataLoader.

    Args:
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Maximum number of samples to load.

    Returns:
        tuple: (test_loader, test_ids)
    """
    # Load Test Data
    test_data = load_and_cache_data(
        Config.TEST_PARQUET,
        "test_data",
        is_test=True,
        load_cached_data=load_cached_data,
    )
    test_ids, test_seq, test_loop, test_dist = test_data

    # Debugging: Slice dataset if max_samples is specified
    if max_samples is not None:
        test_ids = test_ids[:max_samples]
        test_seq = test_seq[:max_samples]
        test_loop = test_loop[:max_samples]
        test_dist = test_dist[:max_samples]

    # Create Dataset
    # For test, RNADataset expects: sequences, loops, distances (no targets)
    test_ds = RNADataset(test_seq, test_loop, test_dist)

    # Create DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader, test_ids
