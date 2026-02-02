import torch
from torch.utils.data import DataLoader
from library.config import Config, RNADataset

# The RNADataset class is imported from library.config to avoid re-implementation.
# library.config.RNADataset already encapsulates:
# 1. Data loading from CSVs (train/val/test).
# 2. Caching logic (checking/saving .npz files).
# 3. Preprocessing logic, specifically 'get_structure_indices' which generates
#    the partner, partner-1, and partner+1 index maps required for the
#    Dense Latent-Neighbor Hybrid Network (Idea 14).


def get_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates and returns DataLoaders for the training and validation sets.

    Args:
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Instantiate datasets
    # The caching mechanism is handled internally by RNADataset -> get_dataset
    train_dataset = RNADataset("train")
    val_dataset = RNADataset("val")

    # Determine if pin_memory should be used (beneficial for GPU training)
    use_pin_memory = Config.DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates and returns the DataLoader for the test set.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: The test data loader.
    """
    test_dataset = RNADataset("test")

    use_pin_memory = Config.DEVICE.type == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return test_loader
