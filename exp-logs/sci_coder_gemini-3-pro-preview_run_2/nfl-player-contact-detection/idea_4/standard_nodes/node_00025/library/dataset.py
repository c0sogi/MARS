import torch
import numpy as np
from torch.utils.data import Dataset


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.

    This dataset wraps the flattened wide feature vectors and provides the specific
    inputs required by the Wide-Input Time-Resolved Gated Network (WI-TRGN),
    specifically the 'is_ground' indicator used for the hard gating mechanism.
    """

    def __init__(self, features, targets=None, is_ground=None):
        """
        Args:
            features (np.ndarray): Scaled feature matrix of shape (N, Input_Dim).
            targets (np.ndarray, optional): Binary target array of shape (N,).
                                            Should be None for inference.
            is_ground (np.ndarray, optional): Binary ground indicator array of shape (N,).
                                              1.0 indicates the pair involves the ground.
        """
        # Convert features to FloatTensor
        self.features = torch.as_tensor(features, dtype=torch.float32)

        # Handle targets
        if targets is not None:
            self.targets = torch.as_tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        # Handle is_ground indicator
        if is_ground is not None:
            self.is_ground = torch.as_tensor(is_ground, dtype=torch.float32)
        else:
            # Default to zeros (Player-Player) if not provided
            self.is_ground = torch.zeros(len(self.features), dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Yields a single sample.

        Returns:
            x (torch.FloatTensor): The feature vector.
            y (torch.FloatTensor): The target label (0.0 if targets are None).
            g (torch.FloatTensor): The is_ground indicator.
        """
        x = self.features[idx]
        g = self.is_ground[idx]

        if self.targets is not None:
            y = self.targets[idx]
            return x, y, g
        else:
            # Return dummy target for inference consistency
            return x, torch.tensor(0.0, dtype=torch.float32), g
