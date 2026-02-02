import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from library.config import Config
from library.model import load_and_process_data, ShipIcebergDataset

# Alias the library dataset to match the requested class name
IcebergDataset = ShipIcebergDataset


def get_fold_loaders(
    train_idx, val_idx, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates training and validation DataLoaders for a specific fold with leak-free angle imputation.

    Args:
        train_idx (np.array): Indices for the training subset.
        val_idx (np.array): Indices for the validation subset.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load all processed data (uses caching internally)
    X_train, y_train, angles_train, ids_train, _, _, _ = load_and_process_data(
        load_cached_data=True
    )

    # Subset the data based on the fold indices
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    ang_tr = angles_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    ang_val = angles_train[val_idx]

    # --- Leak-Free Imputation Logic ---
    # 1. Identify valid angles in the training subset ONLY
    valid_angles = ang_tr[~np.isnan(ang_tr)]

    # 2. Calculate median from training data
    if len(valid_angles) > 0:
        median_angle = np.median(valid_angles)
    else:
        median_angle = 0.0  # Fallback if no valid angles exist (unlikely)

    # 3. Impute missing values in training data
    ang_tr_imputed = np.where(np.isnan(ang_tr), median_angle, ang_tr)

    # 4. Impute missing values in validation data using the TRAINING median
    ang_val_imputed = np.where(np.isnan(ang_val), median_angle, ang_val)

    # --- Augmentations ---
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])

    # --- Dataset Creation ---
    train_ds = IcebergDataset(X_tr, y_tr, ang_tr_imputed, transform=train_transform)
    val_ds = IcebergDataset(X_val, y_val, ang_val_imputed, transform=None)

    # --- DataLoader Creation ---
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


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates a DataLoader for the test set.
    Imputes missing incidence angles using the median of the full training set.

    Args:
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (test_loader, ids_test)
    """
    # Load data
    X_train, _, angles_train, _, X_test, angles_test, ids_test = load_and_process_data(
        load_cached_data=True
    )

    # Calculate global training median for test set imputation
    valid_angles = angles_train[~np.isnan(angles_train)]
    if len(valid_angles) > 0:
        median_angle = np.median(valid_angles)
    else:
        median_angle = 0.0

    # Impute test angles
    ang_test_imputed = np.where(np.isnan(angles_test), median_angle, angles_test)

    # Create Dataset (no targets, no transforms)
    test_ds = IcebergDataset(X_test, None, ang_test_imputed, transform=None)

    # Create DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, ids_test
