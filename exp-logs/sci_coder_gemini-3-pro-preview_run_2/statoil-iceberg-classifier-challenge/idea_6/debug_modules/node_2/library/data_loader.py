import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import process_data, IcebergDataset


def load_and_process_data(load_cached_data=True, base_dir="./working/idea_6"):
    """
    Loads and processes the dataset, handling caching, imputation, and scaling.
    Wraps the library.utils.process_data function to ensure consistency.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.
        base_dir (str): Directory for storing/loading cached data.

    Returns:
        tuple: (X_train, y_train, inc_train, X_test, inc_test, test_ids)
    """
    # Ensure the directory exists as per requirements
    os.makedirs(base_dir, exist_ok=True)

    # Delegate to the provided utility function
    return process_data(load_cached_data=load_cached_data, base_dir=base_dir)


def get_kfold_loaders(X, y, inc, n_splits=5, batch_size=32, seed=42, num_workers=2):
    """
    Generates Stratified K-Fold DataLoaders for the ensemble training pipeline.

    Args:
        X (np.ndarray): Training images (N, 3, 75, 75).
        y (np.ndarray): Training labels (N,).
        inc (np.ndarray): Training incidence angles (N,).
        n_splits (int): Number of cross-validation folds.
        batch_size (int): Batch size for the loaders.
        seed (int): Random seed for the split.
        num_workers (int): Number of subprocesses for data loading.

    Yields:
        tuple: (train_loader, val_loader) for each fold.
    """
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Iterate through folds
    for train_idx, val_idx in skf.split(X, y):
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

        # Create Datasets
        # Enable augmentation (transform=True) only for training
        train_ds = IcebergDataset(
            X_train_fold, inc_train_fold, y_train_fold, transform=True
        )
        val_ds = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

        # Create DataLoaders
        # Use pin_memory=True for faster transfer to GPU
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        yield train_loader, val_loader


def get_test_loader(X_test, inc_test, batch_size=64, num_workers=2):
    """
    Creates a DataLoader for the test set inference.

    Args:
        X_test (np.ndarray): Test images.
        inc_test (np.ndarray): Test incidence angles.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: The loader for the test set.
    """
    test_ds = IcebergDataset(X_test, inc_test, y=None, transform=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader
