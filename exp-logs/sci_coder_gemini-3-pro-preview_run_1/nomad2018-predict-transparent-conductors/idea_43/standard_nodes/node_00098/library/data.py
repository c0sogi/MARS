import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import load_and_preprocess_data


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material data.
    Wraps the dictionary structure returned by the feature processing pipeline.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict: Dictionary containing:
                - atomic_features: (N_total_atoms, n_atom_feats) numpy array
                - batch_indices: (N_total_atoms,) numpy array mapping atoms to crystal index
                - global_features: (N_crystals, n_global_feats) numpy array
                - targets: (N_crystals, 2) numpy array
                - ids: (N_crystals,) numpy array
        """
        self.ids = data_dict["ids"]
        self.global_features = torch.tensor(
            data_dict["global_features"], dtype=torch.float32
        )
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)

        # Efficiently split the flat atomic_features array into a list of tensors per crystal.
        # The batch_indices array is assumed to be sorted and contiguous (0,0,0, 1,1, ..., N-1,N-1).
        # We use np.bincount to find the number of atoms per crystal and then np.split.

        atomic_feats_np = data_dict["atomic_features"]
        batch_idx_np = data_dict["batch_indices"]

        # Count atoms per crystal ID (indices are 0 to len(ids)-1)
        # minlength ensures we account for the last crystal even if it's the last index
        counts = np.bincount(batch_idx_np, minlength=len(self.ids))

        # Calculate split indices (cumulative sum of counts)
        # We exclude the last element because np.split expects N-1 split points for N sections
        splits = np.cumsum(counts)[:-1]

        # Split the array
        atomic_arrays = np.split(atomic_feats_np, splits)

        # Convert to list of tensors
        self.atomic_features_list = [
            torch.tensor(arr, dtype=torch.float32) for arr in atomic_arrays
        ]

        # Verification
        assert len(self.atomic_features_list) == len(
            self.ids
        ), f"Mismatch between split atomic features ({len(self.atomic_features_list)}) and number of crystals ({len(self.ids)})"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary for a single crystal.
        """
        return {
            "atomic_features": self.atomic_features_list[idx],
            "global_features": self.global_features[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def sparse_collate_fn(batch):
    """
    Collate function for sparse batching (Deep Sets / Graph-like structure).

    Args:
        batch: List of dictionaries from MaterialDataset.__getitem__

    Returns:
        Dictionary with batched tensors:
            - atomic_features: (Total_Atoms_In_Batch, Atom_Feat_Dim)
            - batch_indices: (Total_Atoms_In_Batch,) LongTensor mapping atoms to batch element index
            - global_features: (Batch_Size, Global_Feat_Dim)
            - targets: (Batch_Size, 2)
            - ids: List of IDs
    """
    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []
    batch_indices_list = []

    for i, sample in enumerate(batch):
        atoms = sample["atomic_features"]
        n_atoms = atoms.shape[0]

        atomic_feats_list.append(atoms)
        # Create an index vector [i, i, ..., i] for the current sample
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_feats_list.append(sample["global_features"])
        targets_list.append(sample["targets"])
        ids_list.append(sample["id"])

    # Concatenate atomic features along the 0-th dimension (flattening the batch)
    atomic_features_batch = torch.cat(atomic_feats_list, dim=0)

    # Concatenate batch indices
    batch_indices_batch = torch.cat(batch_indices_list, dim=0)

    # Stack global features and targets (standard batching)
    global_features_batch = torch.stack(global_feats_list, dim=0)
    targets_batch = torch.stack(targets_list, dim=0)

    return {
        "atomic_features": atomic_features_batch,
        "batch_indices": batch_indices_batch,
        "global_features": global_features_batch,
        "targets": targets_batch,
        "ids": ids_list,
    }


def get_data_loaders(batch_size=Config.BATCH_SIZE, load_cached=True, num_workers=2):
    """
    Loads processed data and returns PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for training/inference.
        load_cached (bool): Whether to try loading from cache first.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load and Preprocess Data (using library function)
    # This handles feature extraction, scaling, and caching.
    train_dict, val_dict, test_dict = load_and_preprocess_data(load_cached=load_cached)

    # 2. Create Datasets
    train_dataset = MaterialDataset(train_dict)
    val_dataset = MaterialDataset(val_dict)
    test_dataset = MaterialDataset(test_dict)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
