import os
import torch
from torch.utils.data import DataLoader
from library.utils import process_data, ForestCoverDataset


def get_dataloaders(
    batch_size=4096,
    load_cached_data=True,
    cache_dir="./working/idea_10/",
    quick_run=False,
):
    """
    Orchestrates the data loading pipeline for the Forest Cover Type task.

    Delegates feature engineering and preprocessing to library.utils.process_data,
    then wraps the results in PyTorch DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        cache_dir (str): Directory to store/load cached data.
        quick_run (bool): If True, subsets the data for rapid debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """

    # Delegate to the provided utility function which handles:
    # 1. Metadata loading (train/val/test parquet)
    # 2. Physics-informed feature engineering (Aspect_Sin, Hydrology_Dist, etc.)
    # 3. Preprocessing (StandardScaler for continuous, raw for binary)
    # 4. Caching logic (checking/saving to cache_dir)
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(
        load_cached_data=load_cached_data, cache_dir=cache_dir
    )

    # Handle debugging subset
    if quick_run:
        print("Quick run enabled: Subsetting data to small sample...")
        train_X = train_X[:10000]
        train_y = train_y[:10000]
        val_X = val_X[:2000]
        val_y = val_y[:2000]
        test_X = test_X[:2000]
        test_ids = test_ids[:2000]

    # Instantiate Datasets
    train_ds = ForestCoverDataset(train_X, train_y)
    val_ds = ForestCoverDataset(val_X, val_y)
    test_ds = ForestCoverDataset(test_X)

    # Instantiate DataLoaders
    # Using 4 workers and pinned memory for optimal throughput on the available hardware
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, test_ids
