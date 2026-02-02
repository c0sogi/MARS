import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from library.config import Config
from library.feature_engineering import process_dataset
from library.vision_processing import process_vision_dataset


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for the Vision Branch (Spectrograms).
    Wraps pre-computed numpy tensors for efficient access during training.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Input images of shape (N, C, H, W).
            y (np.ndarray, optional): Targets of shape (N,).
            ids (np.ndarray, optional): Segment IDs of shape (N,).
        """
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        # Input X is float32, shape (C, H, W)
        img = torch.from_numpy(self.X[idx])

        if self.y is not None:
            # Return image and target
            # Ensure target is float32 for regression loss
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, target
        else:
            # Return image only (for inference)
            return img


def get_tabular_data(dataset_type: str = "train", load_cached_data: bool = True):
    """
    Retrieves the tabular data (Branch A) using the feature engineering library.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet/npy files.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray or None): Target array.
    """
    return process_dataset(dataset_type=dataset_type, load_cached_data=load_cached_data)


def get_vision_data(dataset_type: str = "train", load_cached_data: bool = True):
    """
    Retrieves the vision data (Branch B) using the vision processing library.
    Applies Log-Scaling (np.log1p) to targets if they exist, as per the strategy.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached npy files.

    Returns:
        X (np.ndarray): Image tensor.
        y (np.ndarray or None): Transformed target array.
        ids (np.ndarray): Segment IDs.
    """
    X, y, ids = process_vision_dataset(
        dataset_type=dataset_type, load_cached_data=load_cached_data
    )

    # Apply Target Scaling for Vision Branch
    if y is not None:
        y = np.log1p(y)

    return X, y, ids


def get_dataloaders(
    batch_size: int = Config.CNN_TRAIN_PARAMS["batch_size"],
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Creates PyTorch DataLoaders for training and validation (Vision Branch).

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
    """
    # Load Training Data
    X_train, y_train, ids_train = get_vision_data("train", load_cached_data)
    train_ds = VolcanoDataset(X_train, y_train, ids_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Validation Data
    X_val, y_val, ids_val = get_vision_data("val", load_cached_data)
    val_ds = VolcanoDataset(X_val, y_val, ids_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(
    batch_size: int = Config.CNN_TRAIN_PARAMS["batch_size"],
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Creates a PyTorch DataLoader for the test set (Vision Branch).

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        test_loader (DataLoader): Test dataloader.
        ids (np.ndarray): Segment IDs corresponding to the data order.
    """
    X_test, y_test, ids_test = get_vision_data("test", load_cached_data)

    # y_test is None, so dataset will return only images
    test_ds = VolcanoDataset(X_test, y=None, ids=ids_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, ids_test
