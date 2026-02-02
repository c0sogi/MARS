import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.features import load_and_process_data

# Set fixed random seeds
torch.manual_seed(42)
np.random.seed(42)


class SelectiveScaler:
    """
    Manages scaling statistics for the dataset.
    Loads the scalers generated during the preprocessing phase in features.py.
    """

    def __init__(self, scaler_path="./working/idea_47/scalers.npz"):
        self.scaler_path = scaler_path
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None
        self._load_scalers()

    def _load_scalers(self):
        if os.path.exists(self.scaler_path):
            data = np.load(self.scaler_path)
            self.atomic_mean = data["atomic_mean"]
            self.atomic_std = data["atomic_std"]
            self.global_mean = data["global_mean"]
            self.global_std = data["global_std"]
        else:
            # Scalers might not exist if train data hasn't been processed yet
            pass

    def unscale_global(self, global_features):
        if self.global_mean is None:
            return global_features
        return global_features * self.global_std + self.global_mean


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material crystals.
    Loads processed data from cache or processes it from scratch using library.features.
    """

    def __init__(
        self,
        split,
        input_dir="./input",
        metadata_dir="./metadata",
        load_cached_data=True,
        debug_size=None,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            input_dir (str): Path to raw input files.
            metadata_dir (str): Path to metadata CSVs.
            load_cached_data (bool): Whether to try loading from cache.
            debug_size (int, optional): If set, limits the dataset size for debugging.
        """
        self.split = split

        # Load data using the provided library function
        # This handles caching, parsing, feature extraction, and scaling
        data = load_and_process_data(
            split_name=split,
            input_dir=input_dir,
            metadata_dir=metadata_dir,
            load_cached_data=load_cached_data,
        )

        self.atomic_features = data["atomic_features"]
        self.global_features = data["global_features"]
        self.targets = data["targets"]
        self.ids = data["ids"]
        self.batch_indices_flat = data["batch_indices"]

        # Pre-calculate slice indices for fast __getitem__
        # batch_indices_flat maps every atom to its crystal index (0 to N-1)
        # We assume the data from process_dataset is ordered by crystal ID row
        # Using bincount is safe because indices are contiguous integers 0..N-1
        if len(self.ids) > 0:
            self.atom_counts = np.bincount(
                self.batch_indices_flat, minlength=len(self.ids)
            )
            self.atom_offsets = np.concatenate(([0], np.cumsum(self.atom_counts)))
        else:
            self.atom_counts = np.array([])
            self.atom_offsets = np.array([0])

        # Debug mode: truncate dataset
        if debug_size is not None and debug_size < len(self.ids):
            self.ids = self.ids[:debug_size]
            self.global_features = self.global_features[:debug_size]
            self.targets = self.targets[:debug_size]
            # Adjust atomic features
            end_atom_idx = self.atom_offsets[debug_size]
            self.atomic_features = self.atomic_features[:end_atom_idx]
            self.atom_offsets = self.atom_offsets[: debug_size + 1]
            self.atom_counts = self.atom_counts[:debug_size]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve atomic features for this crystal
        start = self.atom_offsets[idx]
        end = self.atom_offsets[idx + 1]

        # Shape: (n_atoms_in_crystal, 21)
        atom_feats = torch.from_numpy(self.atomic_features[start:end])

        # Shape: (22,)
        global_feats = torch.from_numpy(self.global_features[idx])

        # Shape: (2,)
        target = torch.from_numpy(self.targets[idx])

        # Scalar ID
        crystal_id = self.ids[idx]

        return atom_feats, global_feats, target, crystal_id


class SparseCollate:
    """
    Collates a list of variable-size crystal samples into a single sparse batch.
    Generates a batch index vector for aggregation (pooling).
    """

    def __call__(self, batch):
        # batch is a list of tuples: (atom_feats, global_feats, target, id)

        atomic_feats_list = []
        global_feats_list = []
        targets_list = []
        ids_list = []
        batch_indices_list = []

        for i, (atoms, globs, target, cid) in enumerate(batch):
            n_atoms = atoms.shape[0]

            atomic_feats_list.append(atoms)
            global_feats_list.append(globs)
            targets_list.append(target)
            ids_list.append(cid)

            # Create batch index for this crystal (repeats 'i' for each atom)
            # Shape: (n_atoms,)
            batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        # Concatenate all to form the batch
        # Shape: (Total_Atoms_in_Batch, 21)
        batch_atomic_feats = torch.cat(atomic_feats_list, dim=0)

        # Shape: (Total_Atoms_in_Batch,)
        batch_indices = torch.cat(batch_indices_list, dim=0)

        # Shape: (Batch_Size, 22)
        batch_global_feats = torch.stack(global_feats_list, dim=0)

        # Shape: (Batch_Size, 2)
        batch_targets = torch.stack(targets_list, dim=0)

        # Shape: (Batch_Size,)
        batch_ids = torch.tensor(ids_list, dtype=torch.long)

        return (
            batch_atomic_feats,
            batch_indices,
            batch_global_feats,
            batch_targets,
            batch_ids,
        )


def get_train_val_loaders(batch_size=32, num_workers=2, debug_size=None):
    """
    Creates DataLoaders for training and validation sets.
    """
    train_dataset = MaterialDataset(
        split="train", load_cached_data=True, debug_size=debug_size
    )

    val_dataset = MaterialDataset(
        split="val", load_cached_data=True, debug_size=debug_size
    )

    collate_fn = SparseCollate()

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

    return train_loader, val_loader


def get_test_loader(batch_size=32, num_workers=2):
    """
    Creates DataLoader for the test set.
    """
    test_dataset = MaterialDataset(split="test", load_cached_data=True)

    collate_fn = SparseCollate()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return test_loader
