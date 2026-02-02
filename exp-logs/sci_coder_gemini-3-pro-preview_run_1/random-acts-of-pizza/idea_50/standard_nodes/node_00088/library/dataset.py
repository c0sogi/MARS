import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PizzaDataset(Dataset):
    """
    A PyTorch Dataset for the Pizza Request Prediction task.

    It handles the dictionary of feature tensors produced by the FeatureEngineer
    and pairs them with target labels if available.
    """

    def __init__(self, features_dict, labels=None):
        """
        Args:
            features_dict (dict): A dictionary where keys are feature names (e.g., 'title_emb',
                                  'control_features') and values are PyTorch tensors or compatible arrays.
                                  All tensors must have the same size in the first dimension (N).
            labels (array-like, optional): Target labels corresponding to the features.
                                           Can be a list, numpy array, pandas Series, or torch Tensor.
        """
        self.features_dict = features_dict

        # Verify that the features dictionary is not empty
        if not self.features_dict:
            raise ValueError("features_dict cannot be empty.")

        # Determine the dataset length from the first tensor in the dictionary
        # and ensure all feature tensors have the same length
        keys = list(self.features_dict.keys())
        self.length = len(self.features_dict[keys[0]])

        for k in keys[1:]:
            if len(self.features_dict[k]) != self.length:
                raise ValueError(
                    f"Dimension mismatch for feature '{k}'. "
                    f"Expected {self.length}, got {len(self.features_dict[k])}."
                )

        # Process labels
        self.labels = labels
        if self.labels is not None:
            # Convert to tensor if necessary
            if isinstance(self.labels, (pd.Series, np.ndarray, list)):
                self.labels = torch.tensor(self.labels, dtype=torch.float32)
            elif isinstance(self.labels, torch.Tensor):
                self.labels = self.labels.float()

            # Verify label length matches features
            if len(self.labels) != self.length:
                raise ValueError(
                    f"Length mismatch: Features have {self.length} samples, "
                    f"but Labels have {len(self.labels)} samples."
                )

    def __len__(self):
        """Returns the total number of samples."""
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (inputs, label) if labels are present.
            dict: inputs if labels are not present.

            'inputs' is a dictionary containing the sliced tensors for the specific index.
        """
        # Slice each feature tensor at the given index
        inputs = {key: tensor[idx] for key, tensor in self.features_dict.items()}

        if self.labels is not None:
            return inputs, self.labels[idx]

        return inputs
