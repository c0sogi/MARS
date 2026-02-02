import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import prepare_data


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Handles Train, Validation, and Test modes by wrapping pre-processed numpy arrays.
    """

    def __init__(self, X, u_out, y=None, ids=None, is_test=False):
        """
        Args:
            X (np.ndarray): Feature array of shape (N, 80, Features).
            u_out (np.ndarray): Control input array of shape (N, 80).
            y (np.ndarray, optional): Target pressure array of shape (N, 80).
            ids (np.ndarray, optional): ID array of shape (N, 80).
            is_test (bool): Flag to indicate test mode.
        """
        # Convert inputs to PyTorch tensors
        # Features and targets are float32
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.is_test = is_test

        if not self.is_test:
            if y is None:
                raise ValueError(
                    "Target 'y' must be provided for training/validation sets."
                )
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            if ids is None:
                raise ValueError("IDs must be provided for the test set.")
            # IDs are integers (long) for submission mapping
            self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for a single breath sequence.
        """
        if self.is_test:
            return {"X": self.X[idx], "u_out": self.u_out[idx], "id": self.ids[idx]}
        else:
            return {"X": self.X[idx], "u_out": self.u_out[idx], "y": self.y[idx]}


def get_data_loaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to load data from cache or recompute.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data using the features library
    # prepare_data handles caching logic and feature engineering internally
    train_data, val_data, test_data = prepare_data(load_cached_data=load_cached_data)

    # Unpack tuples returned by prepare_data
    # Structure: (X, y, u_out) for train/val, (X, ids, u_out) for test
    train_X, train_y, train_uout = train_data
    val_X, val_y, val_uout = val_data
    test_X, test_ids, test_uout = test_data

    # Initialize Datasets
    train_dataset = VentilatorDataset(train_X, train_uout, y=train_y, is_test=False)
    val_dataset = VentilatorDataset(val_X, val_uout, y=val_y, is_test=False)
    test_dataset = VentilatorDataset(test_X, test_uout, ids=test_ids, is_test=True)

    # Initialize DataLoaders
    # Train: Shuffle=True, Drop_Last=True to ensure consistent batch sizes for stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val: Shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test: Shuffle=False
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
