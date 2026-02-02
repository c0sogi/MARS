import torch
from torch.utils.data import Dataset
import numpy as np
from library.features import prepare_dataset


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps the feature engineering pipeline and provides tensors for training/inference.
    """

    def __init__(self, split: str = "train", load_cached_data: bool = True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load pre-processed .npy files from cache.
        """
        super().__init__()
        self.split = split

        # Load processed data using the library function
        # This handles loading raw CSVs, feature engineering, scaling, and caching
        data = prepare_dataset(split=split, load_cached_data=load_cached_data)

        # Extract numpy arrays
        # X shape: (N_breaths, 80, Num_Features)
        # u_out shape: (N_breaths, 80)
        # y shape: (N_breaths, 80) or None
        # ids shape: (N_breaths, 80)
        self.X_np = data["X"]
        self.u_out_np = data["u_out"]
        self.ids_np = data["ids"]
        self.y_np = data["y"]

        # Convert to PyTorch Tensors
        # We use FloatTensor for inputs and targets
        self.X = torch.from_numpy(self.X_np).float()
        self.u_out = torch.from_numpy(self.u_out_np).float()

        # Handle targets
        if self.y_np is not None:
            self.y = torch.from_numpy(self.y_np).float()
        else:
            # For test set, create dummy targets
            self.y = torch.zeros_like(self.u_out).float()

        # IDs are kept as tensors for convenience in the loop, though often not used in gradient
        self.ids = torch.from_numpy(self.ids_np).long()

    def __len__(self):
        """Returns the total number of breaths in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the data for a single breath.

        Returns:
            tuple: (X, u_out, y, id)
                X (Tensor): Input features of shape (80, F)
                u_out (Tensor): Binary mask of shape (80,)
                y (Tensor): Target pressure of shape (80,)
                id (Tensor): Time step IDs of shape (80,)
        """
        return self.X[idx], self.u_out[idx], self.y[idx], self.ids[idx]
