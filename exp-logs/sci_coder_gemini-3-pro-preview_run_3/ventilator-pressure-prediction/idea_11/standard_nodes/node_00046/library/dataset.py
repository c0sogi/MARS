import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import prepare_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Handles the 3D tensor format (N_breaths, Sequence_Length, Features).
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature array of shape (N, 80, F).
            y (np.ndarray, optional): Target array of shape (N, 80).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def prepare_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, scaling, and caching.
    Wraps the library function to ensure consistency with the provided pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy cache.

    Returns:
        Tuple containing (train_x, train_y, val_x, val_y, test_x, test_ids)
    """
    # Delegate to the provided library function which handles:
    # 1. Loading CSVs from metadata
    # 2. generating PID/Lookahead/Physics features
    # 3. RobustScaling
    # 4. Reshaping to (N, 80, F)
    # 5. Caching to ./working/idea_11/
    return prepare_datasets(load_cached_data=load_cached_data)


def get_data_loaders(batch_size=None, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # 1. Prepare Data (Load/Process/Cache)
    train_x, train_y, val_x, val_y, test_x, test_ids = prepare_data(load_cached_data)

    # 2. Handle Debug Mode (Subset data for fast checking)
    if Config.DEBUG:
        print("DEBUG mode enabled: Slicing datasets to 1000 samples.")
        subset_size = 1000
        train_x = train_x[:subset_size]
        train_y = train_y[:subset_size]
        val_x = val_x[:subset_size]
        val_y = val_y[:subset_size]
        test_x = test_x[:subset_size]
        test_ids = test_ids[:subset_size]

    # 3. Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x)

    # 4. Create DataLoaders
    # Pin memory enables faster data transfer to CUDA devices
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batches to maintain stability for BatchNorm/Noise
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

    return train_loader, val_loader, test_loader, test_ids
