import torch
from torch.utils.data import Dataset
import numpy as np


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the EF-WideResNet model.
    Wraps pre-processed wide feature arrays and categorical embeddings.
    """

    def __init__(self, X_continuous, X_categorical, y=None):
        """
        Args:
            X_continuous (np.ndarray or torch.Tensor): Continuous features matrix (N, D).
            X_categorical (dict): Dictionary mapping feature names to tensors or arrays (N,).
            y (np.ndarray or torch.Tensor, optional): Target labels (N,). Defaults to None.
        """
        # Convert continuous features to FloatTensor
        # Performing conversion in __init__ is more efficient for training speed
        if isinstance(X_continuous, np.ndarray):
            self.X_continuous = torch.from_numpy(X_continuous).float()
        else:
            self.X_continuous = X_continuous.float()

        # Process categorical features
        # Ensure they are LongTensors for embedding lookup
        self.X_categorical = {}
        for key, val in X_categorical.items():
            if isinstance(val, np.ndarray):
                self.X_categorical[key] = torch.from_numpy(val).long()
            else:
                self.X_categorical[key] = val.long()

        # Process targets
        self.y = None
        if y is not None:
            if isinstance(y, np.ndarray):
                # Convert to float and reshape to [N, 1] for BCEWithLogitsLoss compatibility
                self.y = torch.from_numpy(y).float().view(-1, 1)
            else:
                self.y = y.float().view(-1, 1)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X_continuous)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - continuous_row (torch.Tensor): Shape (D,)
                - categorical_row (dict): Dictionary of shape {feature_name: scalar_tensor}
                - label (torch.Tensor, optional): Shape (1,)
        """
        # Slice continuous features
        x_cont = self.X_continuous[idx]

        # Slice categorical features
        # Returns a dict of scalar tensors for the specific row
        x_cat = {key: val[idx] for key, val in self.X_categorical.items()}

        if self.y is not None:
            label = self.y[idx]
            return x_cont, x_cat, label
        else:
            return x_cont, x_cat
