import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.preprocessing import load_data
from library.config import BATCH_SIZE


class GNSSWindowDataset(Dataset):
    """
    PyTorch Dataset for GNSS Windowed Data.
    Wraps preprocessed kinematic sequences and environmental context features.
    """

    def __init__(self, split="train", load_cached_data=True, max_samples=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to load preprocessed data from cache.
            max_samples (int, optional): Limit the number of samples for debugging/testing.
        """
        self.split = split

        # Load data using the preprocessor from library
        # Returns: X_kin (N, T, F), X_ctx (N, F), y (N, 2) or None, meta (DataFrame)
        # The caching mechanism is handled within load_data in library.preprocessing
        self.X_kin, self.X_ctx, self.y, self.meta = load_data(split, load_cached_data)

        # Optional subsetting for debugging
        if max_samples is not None:
            self.X_kin = self.X_kin[:max_samples]
            self.X_ctx = self.X_ctx[:max_samples]
            if self.y is not None:
                self.y = self.y[:max_samples]
            self.meta = self.meta.iloc[:max_samples]

        # Convert to PyTorch tensors
        self.X_kin = torch.from_numpy(self.X_kin).float()
        self.X_ctx = torch.from_numpy(self.X_ctx).float()

        if self.y is not None:
            self.y = torch.from_numpy(self.y).float()
        else:
            self.y = None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - kinematic_sequence: (T, F_kin)
            - context_features: (F_ctx)
            - target_residual: (2,) [Only for train/val]
        """
        sample = {
            "kinematic_sequence": self.X_kin[idx],
            "context_features": self.X_ctx[idx],
        }

        if self.y is not None:
            sample["target_residual"] = self.y[idx]

        return sample


def get_dataloader(
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    load_cached_data=True,
    num_workers=2,
    max_samples=None,
):
    """
    Helper function to create a DataLoader for the GNSS dataset.

    Args:
        split (str): Dataset split ('train', 'validation', 'test').
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        load_cached_data (bool): Whether to use cached preprocessed data.
        num_workers (int): Number of worker threads for loading.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = GNSSWindowDataset(
        split=split, load_cached_data=load_cached_data, max_samples=max_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
