import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.features import load_and_process_data
from library.config import TRAINING_PARAMS, WORKING_DIR


class Scaler:
    """
    Utility class to load and hold feature scaling statistics.
    Stats are computed and saved by library.features.load_and_process_data.
    """

    def __init__(self, directory=WORKING_DIR):
        self.stats_path = os.path.join(directory, "scalers.npz")
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None

        if os.path.exists(self.stats_path):
            self._load_stats()

    def _load_stats(self):
        data = np.load(self.stats_path)
        self.atomic_mean = torch.from_numpy(data["atomic_mean"]).float()
        self.atomic_std = torch.from_numpy(data["atomic_std"]).float()
        self.global_mean = torch.from_numpy(data["global_mean"]).float()
        self.global_std = torch.from_numpy(data["global_std"]).float()


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material properties.
    Wraps pre-processed numpy arrays and handles slicing of atomic features.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing processed data arrays
                              (atomic_features, global_features, batch_indices, targets, ids).
        """
        # Convert numpy arrays to tensors
        self.atomic_features = torch.from_numpy(data_dict["atomic_features"]).float()
        self.global_features = torch.from_numpy(data_dict["global_features"]).float()
        self.batch_indices_raw = torch.from_numpy(data_dict["batch_indices"]).long()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.ids = torch.from_numpy(data_dict["ids"]).long()

        # Pre-compute start and end indices for atomic features for each sample
        # batch_indices_raw maps atom -> sample_idx (0 to N_samples-1)
        # We use bincount to get the number of atoms per sample
        # minlength ensures we account for all samples even if the last ones have 0 atoms (unlikely but safe)
        counts = torch.bincount(self.batch_indices_raw, minlength=len(self.ids))

        # Cumulative sum gives the end indices
        cumsum = torch.cumsum(counts, dim=0)

        # Start indices are 0 followed by the cumsum (shifted)
        self.starts = torch.cat((torch.tensor([0], dtype=torch.long), cumsum[:-1]))
        self.ends = cumsum

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Slice atomic features for this sample
        start = self.starts[idx]
        end = self.ends[idx]

        atomic_feats = self.atomic_features[start:end]
        global_feats = self.global_features[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return atomic_feats, global_feats, target, sample_id


def collate_fn(batch):
    """
    Collate function to batch variable-size point clouds.

    Args:
        batch (list): List of tuples (atomic_feats, global_feats, target, sample_id)

    Returns:
        dict: Batched data dictionary
    """
    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []
    batch_indices_list = []

    for i, (atomic, glob, target, sample_id) in enumerate(batch):
        atomic_feats_list.append(atomic)
        global_feats_list.append(glob)
        targets_list.append(target)
        ids_list.append(sample_id)

        # Create batch index vector for this sample: [i, i, ..., i]
        n_atoms = atomic.shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

    # Concatenate all atomic features into one large tensor
    batch_atomic_feats = torch.cat(atomic_feats_list, dim=0)

    # Concatenate batch indices
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # Stack global features and targets
    batch_global_feats = torch.stack(global_feats_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)
    batch_ids = torch.stack(ids_list, dim=0)

    return {
        "atomic_features": batch_atomic_feats,
        "global_features": batch_global_feats,
        "batch_indices": batch_indices,
        "targets": batch_targets,
        "ids": batch_ids,
    }


def get_data_loaders(batch_size=TRAINING_PARAMS["batch_size"], load_cached_data=True):
    """
    Loads processed data and returns DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to load data from cache if available.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load and process data (handles caching internally)
    train_data_dict, val_data_dict, test_data_dict = load_and_process_data(
        load_cached_data=load_cached_data
    )

    # Initialize Datasets
    train_dataset = MaterialDataset(train_data_dict)
    val_dataset = MaterialDataset(val_data_dict)
    test_dataset = MaterialDataset(test_data_dict)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
