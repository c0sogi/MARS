import torch
from torch.utils.data import Dataset
import numpy as np
from library.config import Config
from library.features import get_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Loads preprocessed data using library.features.get_data and converts
    it to PyTorch tensors suitable for the Structure-Shortcut Deep Residual BiGRU model.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from .npz cache or recompute.
        """
        self.split = split

        # Load data using the centralized feature processing function
        # This handles caching, file loading, and raw feature extraction
        data = get_data(split=split, load_cached_data=load_cached_data)

        # Store arrays
        self.ids = data["ids"]
        self.sequence = data["sequence"]  # (N, 107) int32
        self.loop_type = data["loop_type"]  # (N, 107) int32
        self.pair_index = data["pair_index"]  # (N, 107) int64
        self.pair_dist = data["pair_dist"]  # (N, 107) float32

        # Targets are only present for train/val splits
        # Shape: (N, 68, 3)
        self.targets = data.get("targets")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing inputs and targets for a single sample.
        """
        # 1. Prepare Inputs
        # Sequence and Loop Type are categorical indices -> LongTensor
        seq_tensor = torch.tensor(self.sequence[idx], dtype=torch.long)
        loop_tensor = torch.tensor(self.loop_type[idx], dtype=torch.long)

        # Pair Index is used for gathering/indexing -> LongTensor
        pair_index_tensor = torch.tensor(self.pair_index[idx], dtype=torch.long)

        # Pair Distance is a continuous feature -> FloatTensor
        # Unsqueeze to make it (Seq_Len, 1) for concatenation/projection
        pair_dist_tensor = torch.tensor(
            self.pair_dist[idx], dtype=torch.float32
        ).unsqueeze(-1)

        item = {
            "sequence": seq_tensor,
            "loop_type": loop_tensor,
            "pair_index": pair_index_tensor,
            "pair_dist": pair_dist_tensor,
            "id": self.ids[idx],
        }

        # 2. Prepare Targets (if available)
        if self.targets is not None:
            # Targets shape: (68, 3) -> FloatTensor
            # Corresponds to reactivity, deg_Mg_pH10, deg_Mg_50C
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = target_tensor

        return item
