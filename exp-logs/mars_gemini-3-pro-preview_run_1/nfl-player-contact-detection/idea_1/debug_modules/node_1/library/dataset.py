import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Wraps preprocessed feature matrices and target vectors.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (np.ndarray): Feature matrix of shape (N, D).
            targets (np.ndarray, optional): Target vector of shape (N,).
        """
        # Convert features to float32 tensor
        self.features = torch.tensor(features, dtype=torch.float32)

        # Handle targets
        if targets is not None:
            # Convert to float32 and ensure shape is (N, 1) for BCEWithLogitsLoss/BCELoss
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = None

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Returns:
            tuple: (feature_tensor, target_tensor) if targets exist.
            tensor: feature_tensor if targets do not exist.
        """
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


def create_dataloader(
    X,
    y=None,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
):
    """
    Factory function to create a PyTorch DataLoader.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray, optional): Target vector.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data (set True for training).
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to copy tensors into CUDA pinned memory.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = ContactDataset(X, y)

    # Check for CUDA availability to set pin_memory default safely if not explicitly managed
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
