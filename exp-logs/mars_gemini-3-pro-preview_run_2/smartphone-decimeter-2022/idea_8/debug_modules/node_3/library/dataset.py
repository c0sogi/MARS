import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.preprocessor import get_data


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for the Short-Window Relative-State Bi-GRU model.
    Converts preprocessed windowed numpy arrays into PyTorch tensors.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N_samples, Window_Size, Input_Dim).
            y (np.ndarray, optional): Target residuals of shape (N_samples, Output_Dim).
                                      Can be None for inference/testing.
        """
        # Convert numpy arrays to PyTorch tensors immediately for efficiency
        # given the dataset fits comfortably in memory.
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the given index.

        Returns:
            tuple: (input_sequence, target) if y is present
            tensor: input_sequence if y is None
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Factory function to create DataLoaders for train, validation, and test sets.
    Integrates with the library.preprocessor to fetch or compute data.

    Args:
        load_cached_data (bool): Whether to attempt loading data from the cache directory.
                                 If False or if cache is missing, data is reprocessed.
        batch_size (int, optional): Override batch size from Config. Defaults to Config.BATCH_SIZE.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        test_meta (pd.DataFrame): Metadata for the test set (required for submission reconstruction).
    """
    # Use Config batch size if not overridden
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Fetch preprocessed data (numpy arrays)
    train_X, train_y, val_X, val_y, test_X, test_meta = get_data(
        load_cached_data=load_cached_data
    )

    # Initialize PyTorch Datasets
    train_dataset = SequenceDataset(train_X, train_y)
    val_dataset = SequenceDataset(val_X, val_y)
    test_dataset = SequenceDataset(test_X, None)

    # Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer for GPU training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_meta
