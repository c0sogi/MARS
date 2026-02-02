import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.utils import get_data


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type prediction task.
    Wraps numpy arrays for features and labels.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None
        if y is not None:
            # Adjust labels from 1-based (1-7) to 0-based (0-6) for CrossEntropyLoss
            self.y = torch.tensor(y - 1, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def load_and_preprocess_data(
    batch_size=4096,
    load_cached_data=True,
    base_dir="./working/idea_11",
    sample_size=None,
):
    """
    Loads data using the library utility, creates Datasets, and returns DataLoaders.

    Args:
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to load from cache.
        base_dir (str): Directory for caching.
        sample_size (int, optional): Subsample size for debugging.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        test_loader (DataLoader): DataLoader for test set.
        test_ids (np.ndarray): Array of test IDs corresponding to test_loader samples.
    """
    # 1. Get processed numpy arrays from library
    # This handles feature engineering, standardization, and caching
    train_X, train_y, val_X, val_y, test_X, test_ids = get_data(
        load_cached_data=load_cached_data, base_dir=base_dir, sample_size=sample_size
    )

    # 2. Create PyTorch Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X)  # Test set has no labels

    # 3. Create DataLoaders
    # Using 4096 batch size as specified in the "Budgeting" section
    # Pin memory for faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
