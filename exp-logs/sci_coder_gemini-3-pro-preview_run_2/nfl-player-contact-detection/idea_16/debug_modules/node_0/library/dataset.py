import torch
from torch.utils.data import Dataset
import numpy as np


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Ego-Centric Deep Cross Network (EC-DCN).
    Wraps the flattened, pre-processed feature matrix and targets.
    """

    def __init__(self, features, targets=None, limit_size=None):
        """
        Args:
            features (np.ndarray): Flattened feature matrix of shape (N, Input_Dim).
            targets (np.ndarray, optional): Target labels of shape (N,). Defaults to None.
            limit_size (int, optional): If provided, limits the dataset to the first N samples.
                                        Useful for debugging and quick iterations.
        """
        # Apply limit for debugging if requested
        if limit_size is not None:
            features = features[:limit_size]
            if targets is not None:
                targets = targets[:limit_size]

        # Convert features to FloatTensor
        # We do this upfront to avoid overhead during the training loop,
        # as the dataset fits within memory (approx 700MB-1GB for 3.4M rows).
        self.features = torch.tensor(features, dtype=torch.float32)

        self.targets = None
        if targets is not None:
            # Convert targets to FloatTensor for BCEWithLogitsLoss
            # Reshape from (N,) to (N, 1) to match the shape of model output logits
            self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves the feature vector and label (if available) at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple or torch.Tensor: (features, target) if targets exist, else features.
        """
        x = self.features[idx]

        if self.targets is not None:
            y = self.targets[idx]
            return x, y

        return x
