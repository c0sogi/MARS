import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import FeatureEngineer


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Wraps processed numpy arrays and converts them to tensors.
    """

    def __init__(self, data: dict, is_test: bool = False):
        """
        Args:
            data (dict): Dictionary containing 'x', 'mask', and optionally 'y'.
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.x = data["x"]
        self.mask = data["mask"]
        self.is_test = is_test

        if not self.is_test:
            self.y = data["y"]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        """
        Returns:
            tuple: (x, mask, y) if training/val, otherwise (x, mask)
            x: Scaled model features (Stream A)
            mask: Raw binary features (Stream B)
            y: Target pressure
        """
        # Convert to float32 tensors
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)

        if self.is_test:
            return x, mask
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, mask, y


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Orchestrates the data pipeline:
    1. Calls FeatureEngineer to load/process/cache data.
    2. Wraps data in VentilatorDataset.
    3. Returns DataLoaders and Test IDs.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker subprocesses.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # Process Data (Loading, Engineering, Scaling, Caching handled by FE)
    # This respects the load_cached_data flag passed from main
    data = fe.process_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = VentilatorDataset(data["train"], is_test=False)
    val_dataset = VentilatorDataset(data["val"], is_test=False)
    test_dataset = VentilatorDataset(data["test"], is_test=True)

    # Create DataLoaders
    # drop_last=True for train to ensure consistent batch sizes for BatchNorm stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract Test IDs for submission mapping
    # These are flat arrays of shape (N_samples,)
    test_ids = data["test"]["ids"]

    return train_loader, val_loader, test_loader, test_ids
