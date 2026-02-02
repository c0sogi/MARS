import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from library.config import BATCH_SIZE, NUM_WORKERS, SEED, NUM_FOLDS
from library.model import IcebergDataset, load_data


def get_loaders(fold, debug=False):
    """
    Creates training and validation DataLoaders for a specific fold
    using Stratified K-Fold Cross Validation on the full dataset.

    Args:
        fold (int): The fold index (0 to NUM_FOLDS-1).
        debug (bool): If True, uses a subset of data for debugging.

    Returns:
        train_loader (DataLoader): DataLoader for the training subset.
        val_loader (DataLoader): DataLoader for the validation subset.
    """
    # Load raw data and global statistics
    # load_data handles caching of stats and loading of JSONs
    train_data, _, stats, inc_mean = load_data(debug=debug)

    # Extract targets and IDs to perform stratification
    labels = [x["is_iceberg"] for x in train_data]
    ids = [x["id"] for x in train_data]

    # Initialize Stratified K-Fold with fixed seed for reproducibility
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Iterate to find the indices for the specific fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(ids, labels)):
        if i == fold:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold} is out of range. Max folds: {NUM_FOLDS}")

    # Create data subsets
    train_subset = [train_data[i] for i in train_idx]
    val_subset = [train_data[i] for i in val_idx]

    # Initialize Datasets
    # Train: Augmentation Enabled (Rotations/Flips)
    train_ds = IcebergDataset(
        train_subset, stats, augment=True, inc_angle_mean=inc_mean
    )
    # Validation: Augmentation Disabled
    val_ds = IcebergDataset(val_subset, stats, augment=False, inc_angle_mean=inc_mean)

    # Initialize DataLoaders
    # drop_last=True is crucial for Training to prevent Batch Norm errors
    # if the last batch has size 1.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(debug=False):
    """
    Creates a DataLoader for the test set.

    Args:
        debug (bool): If True, uses a subset of data.

    Returns:
        test_loader (DataLoader): DataLoader for the test set.
    """
    # Load data (we only need test_data here)
    _, test_data, stats, inc_mean = load_data(debug=debug)

    # Initialize Dataset (No augmentation for inference)
    test_ds = IcebergDataset(test_data, stats, augment=False, inc_angle_mean=inc_mean)

    # Initialize DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
