import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from library.data_processing import get_data
from library.config import BATCH_SIZE, NUM_WORKERS, SEED, setup_reproducibility

# Ensure reproducibility for any operations in this module
setup_reproducibility(SEED)


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Wraps the processed feature DataFrame and label array.
    """

    def __init__(self, X, y):
        """
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        # Convert DataFrame to numpy if necessary
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)

        # Convert labels to float32 for BCEWithLogitsLoss
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Loads data splits and returns PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading cached processed data.

    Returns:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        test_loader (DataLoader): Loader for test data.
        test_ids (np.ndarray): Contact IDs for the test set (needed for submission).
    """

    # --- Load Data ---
    # We use the provided data_processing library to get X, y, and ids
    X_train, y_train, _ = get_data("train", load_cached_data=load_cached_data)
    X_val, y_val, _ = get_data("validation", load_cached_data=load_cached_data)
    X_test, y_test, test_ids = get_data("test", load_cached_data=load_cached_data)

    # --- Create Datasets ---
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)
    test_dataset = ContactDataset(X_test, y_test)

    # --- Create DataLoaders ---
    # Shuffle training data, but not validation or test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, test_ids
