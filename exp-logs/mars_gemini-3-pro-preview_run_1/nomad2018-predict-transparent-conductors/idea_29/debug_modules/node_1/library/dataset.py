import torch
from torch.utils.data import Dataset
import numpy as np
import os
from library.data_utils import process_dataset, get_scalers, apply_scalers


class MaterialDataset(Dataset):
    def __init__(
        self, metadata_path, scalers=None, load_cached_data=True, max_samples=None
    ):
        """
        PyTorch Dataset for material formation and bandgap energy prediction.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            scalers (dict, optional): Dictionary of scalers (mean/std) for normalization.
                                      If None, scalers are computed from the loaded data.
            load_cached_data (bool): Whether to load pre-processed data from cache.
            max_samples (int, optional): Limit the number of samples for debugging.
        """
        self.metadata_path = metadata_path

        # Load and process data (geometry + tabular)
        # The process_dataset function in data_utils handles caching logic.
        raw_data = process_dataset(metadata_path, load_cached_data=load_cached_data)

        # Handle scaling
        if scalers is None:
            # Compute scalers if not provided (typically for training set)
            self.scalers = get_scalers(raw_data)
        else:
            self.scalers = scalers

        # Apply scaling
        self.data = apply_scalers(raw_data, self.scalers)

        # Extract components
        ids = self.data["ids"]
        global_feats = self.data["global_features"]
        targets = self.data["targets"]
        atomic_feats = self.data["atomic_features"]

        # Handle subsetting for debugging
        if max_samples is not None:
            ids = ids[:max_samples]
            global_feats = global_feats[:max_samples]
            targets = targets[:max_samples]
            atomic_feats = atomic_feats[:max_samples]

        # Convert to PyTorch tensors
        self.ids = ids
        self.global_features = torch.tensor(global_feats, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # Atomic features are variable length, keep as list of tensors
        self.atomic_features = [
            torch.tensor(f, dtype=torch.float32) for f in atomic_feats
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary for a single sample.
        """
        return {
            "atomic_features": self.atomic_features[idx],
            "global_features": self.global_features[idx],
            "target": self.targets[idx],
            "id": self.ids[idx],
        }


def collate_materials(batch):
    """
    Collate function to handle variable number of atoms per crystal by padding.

    Args:
        batch (list): List of samples from MaterialDataset.

    Returns:
        dict: Batch dictionary containing:
            - atomic_features: (B, Max_N, 8)
            - global_features: (B, 15)
            - targets: (B, 2)
            - mask: (B, Max_N)
            - ids: List of IDs
    """
    # Extract items
    atomic_feats_list = [item["atomic_features"] for item in batch]
    global_feats = torch.stack([item["global_features"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    ids = [item["id"] for item in batch]

    # Determine dimensions for padding
    max_atoms = max([f.shape[0] for f in atomic_feats_list])
    batch_size = len(batch)
    feature_dim = atomic_feats_list[0].shape[1]

    # Initialize padded tensor and mask
    padded_atomic = torch.zeros(
        (batch_size, max_atoms, feature_dim), dtype=torch.float32
    )
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    # Fill tensors
    for i, feats in enumerate(atomic_feats_list):
        n_atoms = feats.shape[0]
        padded_atomic[i, :n_atoms, :] = feats
        mask[i, :n_atoms] = True

    return {
        "atomic_features": padded_atomic,
        "global_features": global_feats,
        "targets": targets,
        "mask": mask,
        "ids": ids,
    }
