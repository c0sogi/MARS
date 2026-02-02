import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import get_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps pre-processed numpy arrays and converts them to tensors on demand.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, 80, N_features).
            y (np.ndarray, optional): Target values of shape (N_breaths, 80). Defaults to None.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Returns:
            X_tensor (torch.Tensor): Float32 tensor of shape (80, N_features).
            y_tensor (torch.Tensor, optional): Float32 tensor of shape (80,).
        """
        # Convert numpy array to torch tensor with float32 precision
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return X_tensor, y_tensor

        return X_tensor


def load_and_process_data(load_cached: bool = True) -> dict:
    """
    Orchestrates reading CSVs, applying feature engineering, and caching.
    Delegates the heavy lifting to the FeatureEngineer in library.features.

    Args:
        load_cached (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing processed numpy arrays (train_x, train_y, etc.).
    """
    return get_datasets(load_cached=load_cached)


def get_data_loaders(batch_size: int = Config.BATCH_SIZE, load_cached: bool = True):
    """
    Constructs DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        load_cached (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data (either from cache or by processing raw files)
    data = load_and_process_data(load_cached=load_cached)

    # Extract arrays
    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]

    # Initialize Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x, y=None)

    # Determine if pinned memory should be used (optimization for GPU)
    use_pin_memory = Config.DEVICE == "cuda"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain consistent statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
