import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import prepare_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps preprocessed numpy arrays for features, mask, and targets.
    """

    def __init__(self, X: np.ndarray, u_out: np.ndarray, y: np.ndarray = None):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, Seq_Len, N_features).
            u_out (np.ndarray): Binary mask indicating expiratory phase (N_breaths, Seq_Len).
            y (np.ndarray, optional): Target pressure values (N_breaths, Seq_Len).
        """
        self.X = torch.FloatTensor(X)
        self.u_out = torch.FloatTensor(u_out)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input': Tensor (Seq_Len, N_features),
                'u_out': Tensor (Seq_Len,),
                'target': Tensor (Seq_Len,) or None
            }
        """
        sample = {"input": self.X[idx], "u_out": self.u_out[idx]}

        if self.y is not None:
            sample["target"] = self.y[idx]

        return sample


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Orchestrates data loading, dataset creation, and dataloader instantiation.
    Uses library.features.prepare_datasets for caching and preprocessing.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data using the centralized feature pipeline
    # This handles caching, feature engineering, normalization, and reshaping
    data = prepare_datasets(load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = VentilatorDataset(
        X=data["train_x"], u_out=data["train_u_out"], y=data["train_y"]
    )

    val_dataset = VentilatorDataset(
        X=data["val_x"], u_out=data["val_u_out"], y=data["val_y"]
    )

    test_dataset = VentilatorDataset(X=data["test_x"], u_out=data["test_u_out"], y=None)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
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

    return train_loader, val_loader, test_loader
