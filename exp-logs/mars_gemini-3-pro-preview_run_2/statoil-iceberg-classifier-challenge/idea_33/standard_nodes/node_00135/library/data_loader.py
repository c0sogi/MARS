import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config, load_and_process_data, IcebergDataset


def get_data(debug=False):
    """
    Retrieves processed data using the library function.

    Args:
        debug (bool): If True, limits the dataset size for debugging purposes.

    Returns:
        tuple: (X_train, y_train, inc_angle_train, X_test, inc_angle_test, test_ids)
    """
    limit = 100 if debug else None
    # load_and_process_data handles caching, global scaling, and 3-channel construction
    return load_and_process_data(load_cached_data=True, limit_data=limit)


def get_loaders(
    fold_idx,
    X,
    y,
    inc_angles,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Generates PyTorch DataLoaders for a specific fold in Stratified K-Fold CV.

    Args:
        fold_idx (int): The index of the current fold (0 to N_FOLDS-1).
        X (np.ndarray): Training images.
        y (np.ndarray): Training labels.
        inc_angles (np.ndarray): Training incidence angles.
        batch_size (int): Batch size for the loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure reproducibility with fixed seed
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate through the generator to find the indices for the requested fold
    split_generator = skf.split(X, y)

    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(split_generator):
        if i == fold_idx:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(
            f"Fold index {fold_idx} is out of range for {Config.N_FOLDS} folds."
        )

    # Subset the data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    inc_train_fold, inc_val_fold = inc_angles[train_idx], inc_angles[val_idx]

    # Create Datasets
    # Enable augmentation (transform=True) for training
    train_ds = IcebergDataset(
        X_train_fold, inc_train_fold, y_train_fold, transform=True
    )
    # Disable augmentation (transform=False) for validation
    val_ds = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(
    X_test, inc_test, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Generates a PyTorch DataLoader for the test set.

    Args:
        X_test (np.ndarray): Test images.
        inc_test (np.ndarray): Test incidence angles.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: The test data loader.
    """
    # No augmentation for test set
    test_ds = IcebergDataset(X_test, inc_test, transform=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
