import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.features import prepare_datasets
from library.config import BATCH_SIZE


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material science data.

    Handles atomic features (variable size per sample) and global features (fixed size).
    Data is expected to be preprocessed and scaled.
    """

    def __init__(self, data):
        """
        Args:
            data (dict or NpzFile): Dictionary containing dataset arrays:
                - atomic_features: (Total_Atoms, Atomic_Dim)
                - batch_indices: (Total_Atoms,) mapping atom -> sample_idx
                - global_features: (N_samples, Global_Dim)
                - targets: (N_samples, Output_Dim)
                - ids: (N_samples,)
        """
        super().__init__()

        # Load arrays into CPU memory (efficient for access during training)
        self.atomic_features = torch.from_numpy(data["atomic_features"]).float()
        self.global_features = torch.from_numpy(data["global_features"]).float()
        self.targets = torch.from_numpy(data["targets"]).float()
        self.ids = torch.from_numpy(data["ids"]).long()

        # Pre-calculate slice indices for atomic features
        # batch_indices maps each atom to its sample index (0 to N_samples-1)
        # We assume batch_indices are sorted and contiguous (guaranteed by process_subset)
        batch_indices = data["batch_indices"]

        # Count number of atoms for each sample
        # minlength ensures we account for all samples even if some have 0 atoms (unlikely but safe)
        counts = np.bincount(batch_indices, minlength=len(self.targets))

        # Calculate cumulative sum to get end indices
        self.ends = np.cumsum(counts)
        # Start indices are 0 followed by the ends of previous samples
        self.starts = np.concatenate(([0], self.ends[:-1]))

        self.length = len(self.targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            atom_x (Tensor): Atomic features for this crystal (N_atoms, D_a)
            global_x (Tensor): Global features (D_g,)
            y (Tensor): Targets (2,)
            sample_id (Tensor): ID (1,)
        """
        # Retrieve atomic features using pre-calculated slices
        start = self.starts[idx]
        end = self.ends[idx]

        atom_x = self.atomic_features[start:end]
        global_x = self.global_features[idx]
        y = self.targets[idx]
        sample_id = self.ids[idx]

        return atom_x, global_x, y, sample_id


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms per sample.

    Constructs a batch suitable for Deep Sets / Graph Neural Networks.

    Args:
        batch: List of tuples (atom_x, global_x, y, sample_id)

    Returns:
        batch_atom_x (Tensor): Concatenated atomic features (Total_Batch_Atoms, D_a)
        batch_indices (Tensor): Index tensor mapping atoms to batch element (Total_Batch_Atoms,)
        batch_global_x (Tensor): Stacked global features (B, D_g)
        batch_y (Tensor): Stacked targets (B, 2)
        batch_ids (Tensor): Stacked IDs (B,)
    """
    atom_x_list = []
    batch_indices_list = []
    global_x_list = []
    y_list = []
    id_list = []

    for i, (atom_x, global_x, y, sample_id) in enumerate(batch):
        atom_x_list.append(atom_x)

        # Create batch index vector for this sample (all values = i)
        n_atoms = atom_x.shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_x_list.append(global_x)
        y_list.append(y)
        id_list.append(sample_id)

    # Concatenate atomic features and indices along the first dimension
    batch_atom_x = torch.cat(atom_x_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # Stack global features, targets, and IDs
    batch_global_x = torch.stack(global_x_list, dim=0)
    batch_y = torch.stack(y_list, dim=0)
    batch_ids = torch.stack(id_list, dim=0)

    return batch_atom_x, batch_indices, batch_global_x, batch_y, batch_ids


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True):
    """
    Prepares datasets and creates DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): If True, tries to load preprocessed data from disk.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
        test_loader (DataLoader)
    """
    # Load or compute processed data
    train_data, val_data, test_data, _ = prepare_datasets(
        load_cached_data=load_cached_data
    )

    # Instantiate Datasets
    train_dataset = MaterialDataset(train_data)
    val_dataset = MaterialDataset(val_data)
    test_dataset = MaterialDataset(test_data)

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
