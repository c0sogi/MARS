import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_utils import get_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Wraps pre-processed numpy arrays for inputs, targets, and structural indices.
    """

    def __init__(self, inputs, targets, pair_indices, ids):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, Input_Dim)
            targets (np.ndarray): Shape (N, 107, 5)
            pair_indices (np.ndarray): Shape (N, 107)
            ids (np.ndarray): Shape (N,) - Array of strings
        """
        self.inputs = inputs
        self.targets = targets
        self.pair_indices = pair_indices
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors

        # Inputs: (107, D) Float32
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Pair Indices: (107,) Long - Used for gathering partner features
        pairs = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        # Targets: (107, 5) Float32
        # Note: Tail positions (68-107) are 0.0 for boundary anchoring
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # ID: String
        sample_id = self.ids[idx]

        return x, pairs, y, sample_id


def get_dataloaders(load_cached_data=True):
    """
    Initializes and returns DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.
                                 If False or cache missing, re-computes data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data (uses library.data_utils for processing and caching)
    # The get_data function handles the logic of checking cache vs computing from scratch
    train_data = get_data(mode="train", load_cached_data=load_cached_data)
    val_data = get_data(mode="val", load_cached_data=load_cached_data)
    test_data = get_data(mode="test", load_cached_data=load_cached_data)

    # 2. Instantiate Datasets
    train_dataset = RNADataset(
        inputs=train_data["inputs"],
        targets=train_data["targets"],
        pair_indices=train_data["pair_indices"],
        ids=train_data["ids"],
    )

    val_dataset = RNADataset(
        inputs=val_data["inputs"],
        targets=val_data["targets"],
        pair_indices=val_data["pair_indices"],
        ids=val_data["ids"],
    )

    test_dataset = RNADataset(
        inputs=test_data["inputs"],
        targets=test_data["targets"],
        pair_indices=test_data["pair_indices"],
        ids=test_data["ids"],
    )

    # 3. Instantiate Loaders
    # Pin memory is beneficial for GPU training
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Ensure consistent batch sizes for statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
