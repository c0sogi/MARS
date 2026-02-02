import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.data_utils import load_and_cache_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Loads preprocessed data using the library.data_utils module.
    Each item contains:
        - input: (seq_len, input_channels) tensor
        - pair_index: (seq_len,) tensor mapping indices to their paired index
        - target: (pred_len, output_dim) tensor (only for train/val)
        - id: string identifier
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        super().__init__()
        self.split = split
        self.config = Config()

        # Load data using the provided utility function
        # This handles caching, processing, and loading from metadata
        data_dict = load_and_cache_data(
            self.config, split, load_cached_data=load_cached_data
        )

        # Store raw IDs
        self.ids = data_dict["ids"]

        # Convert inputs to tensors
        # Shape: (N, 107, 14)
        self.inputs = torch.from_numpy(data_dict["inputs"]).float()

        # Shape: (N, 107)
        self.pair_indices = torch.from_numpy(data_dict["pair_indices"]).long()

        # Handle targets if they exist (train/val)
        if "targets" in data_dict:
            # Shape: (N, 68, 5)
            self.targets = torch.from_numpy(data_dict["targets"]).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary for the sample at idx.
        """
        sample = {
            "input": self.inputs[idx],
            "pair_index": self.pair_indices[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        return sample
