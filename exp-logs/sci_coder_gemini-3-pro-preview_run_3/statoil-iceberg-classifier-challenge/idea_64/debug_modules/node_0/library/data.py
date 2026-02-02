import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import load_and_process_data, IcebergDataset, set_seed


def get_transforms(mode="train"):
    """
    Returns the transform configuration for the dataset.
    The IcebergDataset in library.utils implements random horizontal and vertical flips
    when the transform argument is truthy.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        bool: True if augmentation should be applied, False otherwise.
    """
    return mode == "train"


def get_fold_loaders(
    fold_index=0, n_splits=5, batch_size=32, load_cached_data=True, seed=42
):
    """
    Prepares DataLoaders for a specific fold of Stratified K-Fold Cross Validation.
    Implements leak-free incidence angle imputation by calculating the median angle
    only on the training subset.

    Args:
        fold_index (int): The index of the fold to retrieve (0 to n_splits-1).
        n_splits (int): Total number of folds.
        batch_size (int): Batch size for the DataLoaders.
        load_cached_data (bool): Whether to load pre-processed numpy arrays from cache.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, fold_median_angle)
    """
    set_seed(seed)

    # Load processed data (handles caching internally in library.utils)
    data = load_and_process_data(load_cached_data=load_cached_data)

    X = data["X_train"]
    y = data["y_train"]
    angles = data["angle_train"]

    # Generate Stratified K-Fold indices
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_index:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold index {fold_index} out of range for {n_splits} splits.")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train, angle_val = angles[train_idx], angles[val_idx]

    # Leak-Free Imputation: Calculate median only on training data
    valid_angles_train = angle_train[~np.isnan(angle_train)]
    fold_median = np.median(valid_angles_train) if len(valid_angles_train) > 0 else 0.0

    # Instantiate Datasets
    # transform=True enables random flips for training
    train_ds = IcebergDataset(
        X_train,
        y_train,
        angle_train,
        transform=get_transforms("train"),
        angle_impute_val=fold_median,
    )

    val_ds = IcebergDataset(
        X_val,
        y_val,
        angle_val,
        transform=get_transforms("val"),
        angle_impute_val=fold_median,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, fold_median


def get_test_loader(batch_size=32, load_cached_data=True, angle_impute_val=0.0):
    """
    Prepares a DataLoader for the test dataset.

    Args:
        batch_size (int): Batch size for the DataLoader.
        load_cached_data (bool): Whether to load pre-processed numpy arrays from cache.
        angle_impute_val (float): The value to use for filling NaN incidence angles
                                  (usually the median from the training folds).

    Returns:
        tuple: (test_loader, test_ids)
    """
    data = load_and_process_data(load_cached_data=load_cached_data)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    test_ds = IcebergDataset(
        X_test,
        None,
        angle_test,
        transform=get_transforms("test"),
        angle_impute_val=angle_impute_val,
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader, ids_test
