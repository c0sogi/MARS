import torch
import numpy as np
from torch.utils.data import Dataset


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Query Alignment-Gated MLP.

    This dataset handles the conversion of pre-computed numpy features into PyTorch tensors
    and generates the necessary attention masks for variable-length user history sequences.
    It expects the data structure produced by library.feature_engineering.FeatureEngineer.
    """

    def __init__(self, mlp_features, labels=None):
        """
        Args:
            mlp_features (dict): Dictionary containing the following keys:
                - 'title_emb': Numpy array of shape (N, D)
                - 'body_emb': Numpy array of shape (N, D)
                - 'history_emb': Numpy array of shape (N, Max_Len, D)
                - 'metadata': Numpy array of shape (N, M)
            labels (array-like, optional): Target labels of shape (N,). Defaults to None.
        """
        # Convert all feature arrays to PyTorch tensors (Float32)
        self.title_emb = torch.tensor(mlp_features["title_emb"], dtype=torch.float32)
        self.body_emb = torch.tensor(mlp_features["body_emb"], dtype=torch.float32)
        self.history_emb = torch.tensor(
            mlp_features["history_emb"], dtype=torch.float32
        )
        self.metadata = torch.tensor(mlp_features["metadata"], dtype=torch.float32)

        # Handle labels if provided (e.g., for training/validation)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Returns:
            dict: A dictionary containing:
                - 'title_emb': Tensor (D,)
                - 'body_emb': Tensor (D,)
                - 'history_emb': Tensor (Max_Len, D)
                - 'metadata': Tensor (M,)
                - 'history_padding_mask': BoolTensor (Max_Len,) - True indicates padding.
                - 'label': Tensor (1,) - Only if labels were provided.
        """
        # Retrieve feature tensors
        title = self.title_emb[idx]
        body = self.body_emb[idx]
        history = self.history_emb[idx]  # Shape: (Max_Len, D)
        meta = self.metadata[idx]

        # Generate History Padding Mask
        # The FeatureEngineer pads sequences with zero vectors.
        # We create a boolean mask where True indicates a padding position (all zeros).
        # This aligns with torch.nn.MultiheadAttention's key_padding_mask expectation.
        history_padding_mask = (history == 0).all(dim=-1)

        sample = {
            "title_emb": title,
            "body_emb": body,
            "history_emb": history,
            "metadata": meta,
            "history_padding_mask": history_padding_mask,
        }

        # Add label if available
        if self.labels is not None:
            sample["label"] = self.labels[idx]

        return sample
