import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import Config, process_data, RNADataset


def _slice_data_dict(data_dict, max_samples):
    """
    Helper function to slice the data dictionary for debugging/subsampling.
    """
    if max_samples is None:
        return data_dict

    # Determine the number of samples available
    # Assuming 'inputs' key exists and has shape (N, ...)
    if "inputs" not in data_dict:
        return data_dict

    total_samples = len(data_dict["inputs"])

    if max_samples >= total_samples:
        return data_dict

    sliced_dict = {}
    for key, value in data_dict.items():
        # Slice only if it's an array/tensor with the same first dimension
        if hasattr(value, "__len__") and len(value) == total_samples:
            sliced_dict[key] = value[:max_samples]
        else:
            sliced_dict[key] = value

    return sliced_dict


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_train_samples=None,
    max_val_samples=None,
):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_train_samples (int, optional): Maximum number of training samples to use (for debugging).
        max_val_samples (int, optional): Maximum number of validation samples to use (for debugging).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Process/Load Training Data
    train_data = process_data(load_cached_data=load_cached_data, mode="train")
    train_data = _slice_data_dict(train_data, max_train_samples)

    train_dataset = RNADataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Process/Load Validation Data
    val_data = process_data(load_cached_data=load_cached_data, mode="val")
    val_data = _slice_data_dict(val_data, max_val_samples)

    val_dataset = RNADataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns the test DataLoader.

    Args:
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        DataLoader: The test data loader.
    """
    # Process/Load Test Data
    test_data = process_data(load_cached_data=load_cached_data, mode="test")

    # Load Test IDs for submission mapping
    # process_data does not return IDs, so we load them from the metadata file
    test_ids = None
    if os.path.exists(Config.TEST_PATH):
        try:
            test_df = pd.read_parquet(Config.TEST_PATH)
            if "id" in test_df.columns:
                test_ids = test_df["id"].values
        except Exception as e:
            print(f"Error loading test IDs from {Config.TEST_PATH}: {e}")

    test_dataset = RNADataset(test_data, ids=test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
