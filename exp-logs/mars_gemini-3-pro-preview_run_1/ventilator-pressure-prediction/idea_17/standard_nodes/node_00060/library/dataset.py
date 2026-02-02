import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import prepare_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Wraps pre-processed numpy arrays and provides them as tensors.
    """

    def __init__(self, X, u_out, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Feature matrix of shape (N_breaths, 80, N_features).
            u_out (np.ndarray): Control flag of shape (N_breaths, 80).
            y (np.ndarray, optional): Target pressure of shape (N_breaths, 80).
            ids (np.ndarray, optional): Time step IDs of shape (N_breaths, 80).
        """
        # Convert to FloatTensor for model inputs/targets
        self.X = torch.tensor(X, dtype=torch.float32)
        # u_out is binary but used as a multiplicative mask, so Float is convenient
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.ids = torch.tensor(ids, dtype=torch.int64) if ids is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for a single breath.
        """
        sample = {"x": self.X[idx], "u_out": self.u_out[idx]}

        if self.y is not None:
            sample["y"] = self.y[idx]

        if self.ids is not None:
            sample["ids"] = self.ids[idx]

        return sample


def prepare_data(load_cached_data=True):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files
                                 from the working directory based on Config hash.

    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_names)
    """
    # 1. Load/Process Data using the library function (handles caching and feature engineering)
    data = prepare_datasets(load_cached_data=load_cached_data)

    # 2. Create Dataset Objects
    train_dataset = VentilatorDataset(
        X=data["train_X"], u_out=data["train_uout"], y=data["train_y"]
    )

    val_dataset = VentilatorDataset(
        X=data["val_X"], u_out=data["val_uout"], y=data["val_y"]
    )

    test_dataset = VentilatorDataset(
        X=data["test_X"], u_out=data["test_uout"], ids=data["test_ids"]
    )

    # 3. Create DataLoaders
    # Drop last batch in training to maintain consistent batch sizes for stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, data["feature_names"]
