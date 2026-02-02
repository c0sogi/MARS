import torch
from torch.utils.data import Dataset
from library.config import Config
from library.data_utils import load_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Wraps the preprocessing logic from library.data_utils to provide tensors for training/inference.
    """

    def __init__(self, split="train", load_cached_data=True, max_samples=None):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached .npy files via data_utils.
            max_samples (int, optional): If set, truncates the dataset to this many samples for debugging.
        """
        self.split = split

        # Load data using the centralized utility
        # This function handles caching, metadata reading, and feature engineering
        data_dict = load_data(split=split, load_cached_data=load_cached_data)

        # Extract raw numpy arrays
        self.ids = data_dict["ids"]
        features = data_dict["features"]
        adjacency = data_dict["adjacency"]
        targets = data_dict.get("targets", None)

        # Truncate dataset if max_samples is specified (useful for debugging pipeline)
        if max_samples is not None and max_samples < len(self.ids):
            self.ids = self.ids[:max_samples]
            features = features[:max_samples]
            adjacency = adjacency[:max_samples]
            if targets is not None:
                targets = targets[:max_samples]

        # Convert to PyTorch Tensors
        # Features: (N, 107, 14) -> Float32
        self.features = torch.from_numpy(features).float()

        # Adjacency: (N, 107) -> Int64 (Long) for indexing/gather operations
        self.adjacency = torch.from_numpy(adjacency).long()

        # Targets: (N, 68, 5) -> Float32, only exists for train/val
        if targets is not None:
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = None

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample.

        Returns:
            dict: Dictionary containing:
                - 'features': Tensor (107, 14)
                - 'adjacency': Tensor (107,)
                - 'targets': Tensor (68, 5) [only if split != test]
                - 'id': str
        """
        sample = {
            "features": self.features[idx],
            "adjacency": self.adjacency[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        return sample
