import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.features import load_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Wraps the pre-processed numpy arrays for sequences, loop types,
    structure distances, and targets.
    """

    def __init__(self, data, split="train"):
        """
        Args:
            data (dict): Dictionary containing processed data arrays from load_data.
            split (str): 'train', 'val', or 'test'.
        """
        self.seq = data["seq"]
        self.loop = data["loop"]
        self.dist = data["dist"]
        self.ids = data["ids"]
        self.split = split

        # Targets are only available for train and val splits
        # Shape: (N, 68, 3)
        if "targets" in data:
            self.targets = data["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors

        # Sequence and Loop are categorical indices for Embeddings -> LongTensor
        seq = torch.tensor(self.seq[idx], dtype=torch.long)
        loop = torch.tensor(self.loop[idx], dtype=torch.long)

        # Distance is a numerical value for Sinusoidal Encoding -> FloatTensor
        # Although stored as int32, we convert to float for math operations in the model
        dist = torch.tensor(self.dist[idx], dtype=torch.float32)

        item = {"seq": seq, "loop": loop, "dist": dist, "id": self.ids[idx]}

        # Include targets if available (Train/Val)
        if self.targets is not None:
            targets = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = targets

        return item


def get_dataloaders(
    split="train",
    batch_size=Config.BATCH_SIZE,
    shuffle=None,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the data.
                                  Defaults to True for 'train', False otherwise.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_samples (int, optional): Limit the number of samples for debugging/testing.

    Returns:
        DataLoader: PyTorch DataLoader configured for the task.
    """
    # 1. Load data using the library function (handles caching and raw file parsing)
    data = load_data(split=split, load_cached_data=load_cached_data)

    # 2. Debugging: Limit dataset size if max_samples is provided
    if max_samples is not None:
        limit = min(max_samples, len(data["seq"]))
        data["seq"] = data["seq"][:limit]
        data["loop"] = data["loop"][:limit]
        data["dist"] = data["dist"][:limit]
        data["ids"] = data["ids"][:limit]
        if "targets" in data:
            data["targets"] = data["targets"][:limit]

    # 3. Create Dataset instance
    dataset = RNADataset(data, split=split)

    # 4. Determine shuffle behavior if not explicitly provided
    if shuffle is None:
        shuffle = split == "train"

    # 5. Create DataLoader
    # drop_last=True for training to maintain consistent batch sizes for BatchNorm/LayerNorm
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(split == "train"),
    )

    return loader
