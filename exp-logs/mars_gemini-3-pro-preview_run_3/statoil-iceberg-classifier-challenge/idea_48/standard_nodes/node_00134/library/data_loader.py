import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import (
    load_and_process_data,
    IcebergDataset,
    BATCH_SIZE,
    SEED,
    NUM_FOLDS,
)


def get_dataloaders(
    fold_idx, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS, seed=SEED, load_cached=True
):
    """
    Generates train and validation DataLoaders for a specific fold in Stratified K-Fold CV.

    Args:
        fold_idx (int): The index of the fold to retrieve (0 to num_folds-1).
        batch_size (int): Batch size for the dataloaders.
        num_folds (int): Total number of folds.
        seed (int): Random seed for reproducibility.
        load_cached (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load all data using the centralized processing function
    X, angles, y, _, _, _ = load_and_process_data(load_cached_data=load_cached)

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Stratified K-Fold Splitter
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Retrieve indices for the requested fold
    # skf.split returns a generator, so we iterate to find the correct fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(
            f"Fold index {fold_idx} is out of range for {num_folds} folds."
        )

    # Create Datasets
    train_dataset = IcebergDataset(
        X[train_idx], angles[train_idx], y[train_idx], transform=train_transform
    )

    val_dataset = IcebergDataset(
        X[val_idx], angles[val_idx], y[val_idx], transform=None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=BATCH_SIZE, load_cached=True):
    """
    Generates the DataLoader for the test set.

    Args:
        batch_size (int): Batch size for the dataloader.
        load_cached (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (test_loader, test_ids)
    """
    # Load test data
    _, _, _, X_test, angles_test, test_ids = load_and_process_data(
        load_cached_data=load_cached
    )

    # Create Dataset (No transforms for test)
    test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader, test_ids
