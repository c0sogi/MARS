import torch
from torch.utils.data import Dataset
import numpy as np


class SKFDataset(Dataset):
    """
    PyTorch Dataset for the Sky-Conditioned Kinematic Filtering Network (SKF-Net).

    Wraps preprocessed numpy arrays for the kinematic stream (windowed time-series)
    and the sky-context stream (window-level stats).
    """

    def __init__(self, X_seq: np.ndarray, X_sky: np.ndarray, y: np.ndarray = None):
        """
        Initialize the dataset.

        Args:
            X_seq (np.ndarray): Kinematic sequence data of shape (N, Window_Size, Features).
                                Expected to be preprocessed (centered, scaled).
            X_sky (np.ndarray): Sky context data of shape (N, Sky_Features).
                                Expected to be preprocessed (scaled).
            y (np.ndarray, optional): Target residuals of shape (N, 2). Defaults to None.
                                      Represents (DeltaEast, DeltaNorth) in meters.
        """
        # Convert numpy arrays to PyTorch tensors (float32)
        # X_seq input shape: (N, Length, Channels)
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_sky = torch.tensor(X_sky, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.X_seq)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (kinematic_tensor, sky_tensor, target_tensor)
                - kinematic_tensor: Shape (Channels, Window_Size) ready for Conv1d.
                - sky_tensor: Shape (Sky_Features,) for the MLP stream.
                - target_tensor: Shape (2,) containing (DeltaEast, DeltaNorth).
                                 Returns a tensor of zeros if targets are not provided.
        """
        # Kinematic stream: stored as (Window, Features), need (Features, Window) for Conv1d
        # Permute dimensions 0 and 1
        kinematic_tensor = self.X_seq[idx].permute(1, 0)

        # Sky context stream: (Sky_Features,)
        sky_tensor = self.X_sky[idx]

        # Target: (2,)
        if self.y is not None:
            target_tensor = self.y[idx]
        else:
            # Return dummy target for inference consistency (DeltaEast=0, DeltaNorth=0)
            target_tensor = torch.zeros(2, dtype=torch.float32)

        return kinematic_tensor, sky_tensor, target_tensor
