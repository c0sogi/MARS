import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the NFL Contact Detection task.
    Wraps preprocessed numpy arrays for features and targets.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (np.ndarray): Input feature matrix of shape (N, D).
            targets (np.ndarray, optional): Target labels of shape (N,). Defaults to None.
        """
        # Store references to the data.
        # We assume features are already scaled and float32/float64 numpy arrays.
        self.features = features
        self.targets = targets

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Returns:
            torch.Tensor: Feature tensor (float32).
            torch.Tensor (optional): Target tensor (float32), if targets exist.
        """
        # Convert the specific row to a tensor on demand.
        # This is memory efficient and works well with num_workers > 0.
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        if self.targets is not None:
            # Targets are converted to float32 for BCEWithLogitsLoss
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y

        return x


def get_dataloader(
    features,
    targets=None,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create a PyTorch DataLoader.

    Args:
        features (np.ndarray): Input features.
        targets (np.ndarray, optional): Target labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data (True for training).
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        torch.utils.data.DataLoader: Configured data loader.
    """
    dataset = ContactDataset(features, targets)

    # Pin memory enables faster data transfer to CUDA devices
    pin_memory = True if Config.DEVICE == "cuda" else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
