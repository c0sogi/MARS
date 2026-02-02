import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import get_data


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    This dataset stores the entire dataset in memory as PyTorch tensors.
    Given the available RAM (220GB) and the dataset size (~4GB processed),
    this approach is significantly faster than reading from disk or converting
    numpy arrays on the fly.
    """

    def __init__(self, X, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, Seq_Len, Features).
            y (np.ndarray, optional): Target pressure of shape (N_breaths, Seq_Len).
            is_test (bool): If True, the dataset acts in inference mode (no targets).
        """
        # Convert to float32 tensors immediately
        self.X = torch.tensor(X, dtype=torch.float32)
        self.is_test = is_test

        if not self.is_test and y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            If train/val: (features, targets)
            If test: (features)
        """
        if self.is_test:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]


def prepare_data(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Orchestrates the data pipeline: loading, processing, and DataLoader creation.

    This function utilizes the `library.features.get_data` function to handle
    the heavy lifting of feature engineering, robust scaling, and caching.
    It then wraps the processed numpy arrays into PyTorch DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
            - train_loader (DataLoader): Shuffle=True
            - val_loader (DataLoader): Shuffle=False
            - test_loader (DataLoader): Shuffle=False
            - test_ids (np.ndarray): Array of IDs corresponding to the test set (N, 80).
    """
    # 1. Load Data (Delegates to the Feature Engineering Pipeline)
    # This step handles:
    # - Checking cache
    # - Loading raw CSVs (if cache miss)
    # - Computing physical features (Area, Derivatives, etc.)
    # - Applying RobustScaler
    # - Saving to cache (if computed)
    train_x, train_y, val_x, val_y, test_x, test_ids = get_data(
        load_cached_data=load_cached_data
    )

    # 2. Instantiate Datasets
    # Data is converted to Tensors here
    train_dataset = VentilatorDataset(train_x, train_y, is_test=False)
    val_dataset = VentilatorDataset(val_x, val_y, is_test=False)
    test_dataset = VentilatorDataset(test_x, y=None, is_test=True)

    # 3. Create DataLoaders
    # Pin memory enables faster transfer to CUDA devices
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to maintain consistent stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, test_ids
