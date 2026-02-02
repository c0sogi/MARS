import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_utils import load_dataset


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Loads preprocessed data including sequence features, structural adjacency,
    and degradation targets. Handles caching via library.data_utils.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load from .npz cache if available.
        """
        self.split = split

        # Load data using the provided utility
        # This returns a dict with 'ids', 'features', 'adjacency', and optionally 'targets'
        data_dict = load_dataset(split=split, load_cached_data=load_cached_data)

        self.ids = data_dict["ids"]

        # Convert features to FloatTensor (N, 107, 14)
        self.features = torch.from_numpy(data_dict["features"]).float()

        # Process Adjacency for Gather Operations
        # Raw adjacency has -1 for unpaired bases.
        # We need valid indices (0..L-1) for torch.gather.
        # We also need a mask to zero out contributions from unpaired bases (dummy index 0).
        raw_adjacency = torch.from_numpy(data_dict["adjacency"]).long()

        # Create mask: 1 if paired, 0 if unpaired (originally -1)
        self.bpp_mask = (raw_adjacency != -1).float()

        # Replace -1 with 0 to prevent index out of bounds errors during gather.
        # The model must use bpp_mask to ignore the values gathered from index 0
        # when the base is actually unpaired.
        self.adjacency = raw_adjacency.clone()
        self.adjacency[raw_adjacency == -1] = 0

        # Handle Targets
        # Targets are (N, 68, 5) for train/val, and None for test
        if split != "test" and data_dict["targets"] is not None:
            self.targets = torch.from_numpy(data_dict["targets"]).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'features': (107, 14),
                'adjacency': (107,),
                'bpp_mask': (107,),
                'targets': (68, 5) [if available],
                'id': str
            }
        """
        item = {
            "features": self.features[idx],
            "adjacency": self.adjacency[idx],
            "bpp_mask": self.bpp_mask[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        return item
