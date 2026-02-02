import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Stereoscopic Residual-Visual Network (SRV-Net).

    Handles dual-stream input:
    1. Kinematic Stream: Flattened tracking features over a temporal window.
    2. Visual Stream: Stereoscopic helmet features (Sideline vs Endzone) over a temporal window.
    """

    def __init__(
        self,
        X_kin: np.ndarray,
        X_vis: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray = None,
    ):
        """
        Args:
            X_kin (np.ndarray): Kinematic features [N, InputDimKin].
            X_vis (np.ndarray): Visual features [N, InputDimVis].
            y (np.ndarray): Target labels [N].
            ids (np.ndarray, optional): Contact IDs [N]. Useful for tracking during inference.
        """
        # Convert to float32 tensors
        self.X_kin = torch.tensor(X_kin, dtype=torch.float32)
        self.X_vis = torch.tensor(X_vis, dtype=torch.float32)

        # Ensure targets are (N, 1) for BCEWithLogitsLoss
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.ids = ids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        """
        Returns:
            kinematic_features (Tensor): Shape (InputDimKin,)
            visual_features (Tensor): Shape (InputDimVis,)
            label (Tensor): Shape (1,)
        """
        return self.X_kin[idx], self.X_vis[idx], self.y[idx]


def get_dataloader(
    X_kin: np.ndarray,
    X_vis: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray = None,
    batch_size: int = Config.BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = Config.NUM_WORKERS,
) -> DataLoader:
    """
    Factory function to create a DataLoader for the ContactDataset.

    Args:
        X_kin, X_vis, y, ids: Data arrays.
        batch_size: Batch size for the loader.
        shuffle: Whether to shuffle the data (True for training, False for val/test).
        num_workers: Number of subprocesses for data loading.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = ContactDataset(X_kin, X_vis, y, ids)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster transfer to GPU
        drop_last=(
            shuffle and len(dataset) > batch_size
        ),  # Drop incomplete batch only during training
    )
