import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_utils import load_and_cache_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Loads preprocessed data (features, adjacency matrices, targets) using
    the library.data_utils module. Handles conversion to PyTorch tensors.
    """

    def __init__(self, split="train", debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, loads a small subset of data for debugging.
        """
        self.split = split
        self.debug = debug

        # Load data using the provided utility which handles caching
        # This returns a dictionary with numpy arrays
        self.data_dict = load_and_cache_data(split, load_cached_data=True, debug=debug)

        # Unpack data
        self.inputs = self.data_dict["inputs"]  # Shape: (N, 107, 14)
        self.pair_indices = self.data_dict["pair_indices"]  # Shape: (N, 107)
        self.ids = self.data_dict["ids"]  # Shape: (N,)

        # Targets are only present for train/val splits
        self.targets = self.data_dict.get("targets")  # Shape: (N, 68, 5) or None

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'sequence': FloatTensor of shape (107, 14)
                - 'pair_indices': LongTensor of shape (107,)
                - 'id': str
                - 'targets': FloatTensor of shape (68, 5) (if available)
        """
        # Convert inputs to FloatTensor
        sequence_tensor = torch.from_numpy(self.inputs[idx]).float()

        # Convert adjacency indices to LongTensor
        # Note: -1 indicates unpaired, model should handle this (e.g., via masking)
        pair_indices_tensor = torch.from_numpy(self.pair_indices[idx]).long()

        sample = {
            "sequence": sequence_tensor,
            "pair_indices": pair_indices_tensor,
            "id": self.ids[idx],
        }

        # Add targets if they exist (train/val)
        if self.targets is not None:
            targets_tensor = torch.from_numpy(self.targets[idx]).float()
            sample["targets"] = targets_tensor

        return sample
