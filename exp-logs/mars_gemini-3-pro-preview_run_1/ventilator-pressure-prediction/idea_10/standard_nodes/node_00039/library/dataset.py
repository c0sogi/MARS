import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import prepare_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Wraps preprocessed numpy arrays into PyTorch tensors.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Seq_Len, Features).
            y (np.ndarray, optional): Target pressure of shape (N, Seq_Len).
        """
        # Convert to float32 tensors immediately to save conversion time during iteration
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            If y is present: (feature_tensor, target_tensor)
            If y is None: feature_tensor
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    Uses library.features.prepare_datasets to handle caching and preprocessing.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
            - train_loader
            - val_loader
            - test_loader
            - test_ids (flat array of IDs for submission mapping)
    """
    # 1. Load Data
    # prepare_datasets handles checking the cache hash, loading raw CSVs,
    # feature engineering, scaling, and reshaping.
    data = prepare_datasets(load_cached_data=load_cached_data)

    # Unpack the tuple
    train_x, train_y, val_x, val_y, test_x, test_ids, _, _ = data

    # 2. Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x)

    # 3. Create DataLoaders
    # pin_memory=True enables faster data transfer to CUDA devices
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to ensure shape consistency
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
