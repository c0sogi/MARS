import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import pre-defined classes and functions from the library to avoid re-implementation
from library.config import Config, process_data, IcebergDataset
from library.utils import set_seed


def get_processed_data(load_cached_data=True):
    """
    Retrieves processed data, delegating to the library implementation for
    loading, preprocessing, normalization, and caching.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (X_train, y_train, inc_train, X_test, inc_test, test_ids)
    """
    # The library function handles:
    # 1. Loading metadata and JSON
    # 2. Constructing 3-channel images (Band1, Band2, Mean)
    # 3. Imputing incidence angles
    # 4. Global per-channel MinMax scaling
    # 5. Caching to Config.CACHE_PATH
    return process_data(load_cached_data=load_cached_data)


def get_fold_loaders(
    X,
    y,
    inc,
    fold_idx,
    n_folds=Config.N_FOLDS,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Generates PyTorch DataLoaders for a specific fold in Stratified K-Fold CV.

    Args:
        X (np.ndarray): Training images.
        y (np.ndarray): Training labels.
        inc (np.ndarray): Training incidence angles.
        fold_idx (int): Index of the fold to retrieve (0-based).
        n_folds (int): Total number of folds.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker subprocesses.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure deterministic splits
    set_seed(Config.SEED)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # StratifiedKFold.split returns a generator; iterate to find the requested fold
    for current_fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        if current_fold == fold_idx:
            # Slice the data for this fold
            X_train_fold = X[train_index]
            y_train_fold = y[train_index]
            inc_train_fold = inc[train_index]

            X_val_fold = X[val_index]
            y_val_fold = y[val_index]
            inc_val_fold = inc[val_index]

            # Create Datasets
            # Train dataset includes augmentation (transform=True) as defined in IcebergDataset
            train_dataset = IcebergDataset(
                X_train_fold, inc_train_fold, y_train_fold, transform=True
            )

            # Validation dataset is raw (transform=False)
            val_dataset = IcebergDataset(
                X_val_fold, inc_val_fold, y_val_fold, transform=False
            )

            # Create Loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            return train_loader, val_loader

    raise ValueError(f"Fold index {fold_idx} is out of range for {n_folds} folds.")


def get_test_loader(
    X_test, inc_test, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Generates a PyTorch DataLoader for the test set.

    Args:
        X_test (np.ndarray): Test images.
        inc_test (np.ndarray): Test incidence angles.
        batch_size (int): Batch size.
        num_workers (int): Number of worker subprocesses.

    Returns:
        DataLoader: Test data loader.
    """
    test_dataset = IcebergDataset(X_test, inc_test, y=None, transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader
