import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.model import get_data

# Set fixed random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Wraps preprocessed numpy arrays and converts them to PyTorch tensors.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Input features (flattened image + inc_angle).
            y (np.ndarray, optional): Target labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            # Return a tuple to be consistent with TensorDataset behavior
            # expected by the generate_submission function in library.model
            return (self.X[idx],)


def get_data_loaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, max_samples=None
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to try loading data from cache.
        max_samples (int, optional): If set, truncates the dataset to this many samples.
                                     Useful for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # 1. Retrieve processed data using the provided library function
    # This handles JSON loading, resizing, flattening, imputation, scaling, and caching.
    X_train, y_train, X_val, y_val, X_test, test_ids = get_data(
        load_cached_data=load_cached_data
    )

    # 2. Optional: Subsample for debugging purposes
    if max_samples is not None:
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]
        X_test = X_test[:max_samples]
        test_ids = test_ids[:max_samples]

    # 3. Create Dataset instances
    train_dataset = IcebergDataset(X_train, y_train)
    val_dataset = IcebergDataset(X_val, y_val)
    test_dataset = IcebergDataset(X_test)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_ids
