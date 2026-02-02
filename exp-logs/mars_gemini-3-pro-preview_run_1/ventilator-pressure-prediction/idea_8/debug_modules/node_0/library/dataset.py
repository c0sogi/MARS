import torch
from torch.utils.data import Dataset
import numpy as np


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    Expects input data in the shape (N_breaths, 80, N_features).
    Serves (features, target, u_out) tuples.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, is_test: bool = False):
        """
        Args:
            X (np.ndarray): Feature array of shape (N_breaths, 80, N_features).
            y (np.ndarray, optional): Target array of shape (N_breaths, 80).
            is_test (bool): Flag indicating if this is a test dataset (no targets).
        """
        super().__init__()

        # Convert inputs to torch tensors immediately for efficiency
        # The dataset size is small enough (~500MB) to fit in RAM
        self.X = torch.tensor(X, dtype=torch.float32)

        self.is_test = is_test

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            # Create dummy targets for test set to maintain consistent interface
            # Shape: (N_breaths, 80)
            self.y = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.float32)

    def __len__(self):
        """Returns the total number of breaths in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the data for a single breath.

        Returns:
            tuple: (features, target, u_out)
                - features: Tensor of shape (80, N_features)
                - target: Tensor of shape (80,)
                - u_out: Tensor of shape (80,) representing the expiratory valve status
        """
        # Retrieve features and target for the given index
        x = self.X[idx]
        y = self.y[idx]

        # Extract u_out for masking purposes.
        # Based on data_processing.py, u_out is the last column in the feature matrix.
        # x shape is (80, F), so we take all time steps for the last feature.
        u_out = x[:, -1]

        return x, y, u_out
