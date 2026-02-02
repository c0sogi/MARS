import torch
from torch.utils.data import Dataset
import numpy as np


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request Prediction task.

    Expects a dictionary of processed features including SBERT embeddings for
    requests and user history, as well as normalized metadata.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing the following keys:
                - 'req_emb': np.array of shape (N, 384)
                - 'hist_emb': np.array of shape (N, Max_Len, 384)
                - 'meta': np.array of shape (N, D)
                - 'y': np.array of shape (N,) [Optional, for train/val only]
        """
        # Convert inputs to PyTorch tensors
        self.req_emb = torch.tensor(data_dict["req_emb"], dtype=torch.float32)
        self.hist_emb = torch.tensor(data_dict["hist_emb"], dtype=torch.float32)
        self.meta = torch.tensor(data_dict["meta"], dtype=torch.float32)

        # Handle labels if they exist (Train/Val) vs Test
        if "y" in data_dict and data_dict["y"] is not None:
            self.labels = torch.tensor(data_dict["y"], dtype=torch.float32)
        else:
            self.labels = None

        # Generate Attention Mask
        # The history embedding is pre-padded with zeros in features.py.
        # We create a mask where 1 indicates a valid history item and 0 indicates padding.
        # We check if the sum of absolute values in the embedding dimension is > 0.
        # Shape: (N, Max_Len)
        self.mask = (self.hist_emb.abs().sum(dim=2) > 1e-9).float()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.req_emb)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Returns:
            dict: A dictionary containing:
                - 'request_emb': Tensor (384,)
                - 'history_emb': Tensor (Max_Len, 384)
                - 'metadata': Tensor (D,)
                - 'mask': Tensor (Max_Len,)
                - 'label': Tensor (1,) [If available]
        """
        item = {
            "request_emb": self.req_emb[idx],
            "history_emb": self.hist_emb[idx],
            "metadata": self.meta[idx],
            "mask": self.mask[idx],
        }

        if self.labels is not None:
            # Return label as a scalar tensor
            item["label"] = self.labels[idx]

        return item
