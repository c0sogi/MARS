import torch
import numpy as np
from torch.utils.data import Dataset


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps pre-processed numpy arrays for features, targets, and masks.
    """

    def __init__(self, X: np.ndarray, u_out: np.ndarray, y: np.ndarray = None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, seq_len, input_dim).
            u_out (np.ndarray): Expiratory phase mask of shape (num_breaths, seq_len).
                                0 indicates inspiratory phase (scored), 1 indicates expiratory.
            y (np.ndarray, optional): Target pressure of shape (num_breaths, seq_len).
                                      Should be None for the test set.
        """
        super().__init__()
        self.X = X
        self.u_out = u_out
        self.y = y
        self.is_test = y is None

    def __len__(self) -> int:
        """Returns the total number of breaths in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - If train/val: (features, target, u_out)
                - If test: (features, u_out)
                All elements are torch.FloatTensor.
        """
        # Convert numpy slices to PyTorch tensors
        # Using torch.tensor ensures a copy and handles stride issues if any
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        u_out_tensor = torch.tensor(self.u_out[idx], dtype=torch.float32)

        if self.is_test:
            return x_tensor, u_out_tensor
        else:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor, u_out_tensor
