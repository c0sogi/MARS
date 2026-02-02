import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from library.utils import load_data, seed_everything


class DualStreamDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Stream Strided-View 2.5D Network.
    Wraps the pre-processed numpy arrays from library.utils.load_data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, ids: np.ndarray):
        """
        Args:
            X: Numpy array of shape (N, 2, 64, 256, 256).
            y: Numpy array of shape (N,).
            ids: Numpy array of shape (N,).
        """
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data for the index
        # X shape: (2, 64, 256, 256)
        streams = self.X[idx]
        target = self.y[idx]

        # Convert to torch tensors
        # Ensure float32 for image data and target
        streams_tensor = torch.from_numpy(streams).float()
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Split into Even and Odd streams as per DSSV-Net architecture
        # streams_tensor[0] corresponds to Even View
        # streams_tensor[1] corresponds to Odd View
        even_stream = streams_tensor[0]
        odd_stream = streams_tensor[1]

        # Return format: ((input1, input2), label)
        return (even_stream, odd_stream), target_tensor


def get_dataloaders(
    batch_size: int = 16,
    input_dir: str = "./input",
    cache_dir: str = "./working/idea_35/",
    load_cached_data: bool = True,
    limit_size: int = None,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Constructs DataLoaders for train, validation, and test sets.
    Uses library.utils.load_data for robust caching and processing.
    """
    # Ensure reproducibility
    seed_everything(seed)

    # Load Training Data
    X_train, y_train, ids_train = load_data(
        split="train",
        load_cached_data=load_cached_data,
        limit_size=limit_size,
        cache_dir=cache_dir,
        input_dir=input_dir,
    )

    # Load Validation Data
    X_val, y_val, ids_val = load_data(
        split="val",
        load_cached_data=load_cached_data,
        limit_size=limit_size,
        cache_dir=cache_dir,
        input_dir=input_dir,
    )

    # Load Test Data
    X_test, y_test, ids_test = load_data(
        split="test",
        load_cached_data=load_cached_data,
        limit_size=limit_size,
        cache_dir=cache_dir,
        input_dir=input_dir,
    )

    # Create Datasets
    train_dataset = DualStreamDataset(X_train, y_train, ids_train)
    val_dataset = DualStreamDataset(X_val, y_val, ids_val)
    test_dataset = DualStreamDataset(X_test, y_test, ids_test)

    # Create DataLoaders
    # Pin memory is enabled for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches during training for stability
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

    return train_loader, val_loader, test_loader
