import os
import torch
from torch.utils.data import DataLoader, Subset
from library.config import Config
from library.utils import calculate_global_stats
from library.model import IcebergDataset, get_inc_angle_stats


def get_loaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_stats=True,
):
    """
    Constructs and returns PyTorch DataLoaders for the training, validation, and test sets.

    This function:
    1. Calculates or loads global image statistics for scaling.
    2. Calculates incidence angle statistics from the training set for normalization.
    3. Initializes IcebergDataset instances for Train (with augmentation), Validation, and Test.
    4. Optionally subsets the data if debug mode is enabled.
    5. Returns DataLoaders for each split.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses to use for data loading.
        debug (bool): If True, returns loaders with a small subset of data for debugging.
        load_cached_stats (bool): If True, attempts to load global stats from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define paths to metadata and raw data
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")

    train_json_path = os.path.join(Config.INPUT_DIR, "train.json")
    test_json_path = os.path.join(Config.INPUT_DIR, "test.json")

    # Compute or load global statistics for image scaling
    # This ensures consistent scaling across all folds and inference
    global_stats = calculate_global_stats(load_cached_data=load_cached_stats)

    # Compute incidence angle statistics for normalization (using training set)
    inc_angle_stats = get_inc_angle_stats(train_meta_path)

    # Instantiate Datasets
    # Training set gets augmentation (transform=True)
    train_dataset = IcebergDataset(
        metadata_csv=train_meta_path,
        json_file=train_json_path,
        transform=True,
        global_stats=global_stats,
        inc_angle_stats=inc_angle_stats,
    )

    # Validation set (transform=False)
    val_dataset = IcebergDataset(
        metadata_csv=val_meta_path,
        json_file=train_json_path,
        transform=False,
        global_stats=global_stats,
        inc_angle_stats=inc_angle_stats,
    )

    # Test set (transform=False)
    test_dataset = IcebergDataset(
        metadata_csv=test_meta_path,
        json_file=test_json_path,
        transform=False,
        global_stats=global_stats,
        inc_angle_stats=inc_angle_stats,
    )

    # Handle Debug Mode (Subset)
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        # Ensure we don't exceed dataset length
        train_indices = range(min(len(train_dataset), subset_size))
        val_indices = range(min(len(val_dataset), subset_size))
        test_indices = range(min(len(test_dataset), subset_size))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    # Pin memory is beneficial for GPU training
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
