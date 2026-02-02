import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.features import process_data


class PartialAtomicScaler:
    """
    Custom scaler for atomic features.
    Scales only continuous features (indices 4:), leaving one-hot encoding (0:4) untouched.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        # X shape: (N_atoms, 17)
        # Fit only on continuous columns [4:]
        self.scaler.fit(X[:, 4:])
        return self

    def transform(self, X):
        # Scale continuous part
        X_continuous = self.scaler.transform(X[:, 4:])
        # Concatenate unscaled one-hot part with scaled continuous part
        return np.hstack([X[:, :4], X_continuous])

    def save(self, path_prefix):
        # Save mean and scale to reconstruct later
        np.savez(
            f"{path_prefix}_atomic.npz",
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
        )

    def load(self, path_prefix):
        data = np.load(f"{path_prefix}_atomic.npz")
        self.scaler.mean_ = data["mean"]
        self.scaler.scale_ = data["scale"]
        self.scaler.var_ = data["scale"] ** 2  # Reconstruct var_ for completeness
        return self


class GlobalScaler:
    """
    Standard scaler wrapper for global features.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def save(self, path_prefix):
        np.savez(
            f"{path_prefix}_global.npz",
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
        )

    def load(self, path_prefix):
        data = np.load(f"{path_prefix}_global.npz")
        self.scaler.mean_ = data["mean"]
        self.scaler.scale_ = data["scale"]
        self.scaler.var_ = data["scale"] ** 2
        return self


def get_scalers(atomic_features, global_features, save_path_prefix=None):
    """
    Fits scalers on provided data and optionally saves them.
    """
    a_scaler = PartialAtomicScaler()
    a_scaler.fit(atomic_features)

    g_scaler = GlobalScaler()
    g_scaler.fit(global_features)

    if save_path_prefix:
        a_scaler.save(save_path_prefix)
        g_scaler.save(save_path_prefix)

    return a_scaler, g_scaler


def load_scalers(path_prefix):
    """
    Loads scalers from disk.
    """
    a_scaler = PartialAtomicScaler()
    a_scaler.load(path_prefix)

    g_scaler = GlobalScaler()
    g_scaler.load(path_prefix)

    return a_scaler, g_scaler


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.
    Handles loading from cache, scaling, and target transformation.
    """

    def __init__(
        self,
        metadata_path,
        cache_file,
        scalers=None,
        fit_scalers=False,
        load_cached_data=True,
        transform_targets=False,
    ):

        # 1. Process or Load Data
        # Returns stacked arrays for efficiency
        (
            self.atomic_feats,
            self.global_feats,
            self.targets,
            self.ids,
            self.batch_indices,
        ) = process_data(metadata_path, cache_file, load_cached_data=load_cached_data)

        # 2. Handle Scalers
        scaler_prefix = os.path.splitext(Config.SCALER_CACHE_FILE)[0]

        if fit_scalers:
            self.a_scaler, self.g_scaler = get_scalers(
                self.atomic_feats, self.global_feats, save_path_prefix=scaler_prefix
            )
        elif scalers is not None:
            self.a_scaler, self.g_scaler = scalers
        elif os.path.exists(f"{scaler_prefix}_atomic.npz"):
            self.a_scaler, self.g_scaler = load_scalers(scaler_prefix)
        else:
            self.a_scaler, self.g_scaler = None, None

        # 3. Apply Scaling
        if self.a_scaler and Config.SCALE_ATOMIC_CONTINUOUS:
            self.atomic_feats = self.a_scaler.transform(self.atomic_feats)

        if self.g_scaler and Config.SCALE_GLOBAL:
            self.global_feats = self.g_scaler.transform(self.global_feats)

        # 4. Target Transformation (Log1p)
        if transform_targets:
            self.targets = np.log1p(self.targets)

        # 5. Pre-compute slice indices
        # batch_indices contains the sample ID for each atom.
        # We identify where the sample ID changes to determine start/end of each crystal's atoms.
        # This assumes atoms for a single crystal are contiguous in the array (guaranteed by process_data).
        if len(self.atomic_feats) > 0:
            # Find indices where the value changes
            change_points = (
                np.where(self.batch_indices[:-1] != self.batch_indices[1:])[0] + 1
            )
            # Add start (0) and end (total length)
            splits = np.concatenate(([0], change_points, [len(self.atomic_feats)]))
            self.slices = list(zip(splits[:-1], splits[1:]))
        else:
            self.slices = []

        # Verification
        if len(self.slices) != len(self.global_feats):
            # This might happen if a crystal has 0 atoms (impossible) or data corruption
            # Fallback: assume 1-to-1 mapping failure
            print(
                f"Warning: Number of atomic slices ({len(self.slices)}) != Number of global samples ({len(self.global_feats)})"
            )

    def __len__(self):
        return len(self.global_feats)

    def __getitem__(self, idx):
        # Get atomic features for this specific crystal
        start, end = self.slices[idx]
        atoms = self.atomic_feats[start:end]

        # Get global features
        glob = self.global_feats[idx]

        # Get target and ID
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return (
            torch.tensor(atoms, dtype=torch.float32),
            torch.tensor(glob, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(sample_id, dtype=torch.long),
        )


def collate_sparse_batch(batch):
    """
    Collates a list of samples into a sparse batch.
    - atomic_features: Concatenated (Total_Atoms, D_a)
    - global_features: Stacked (B, D_g)
    - batch_indices: Vector mapping each atom to its batch index (0..B-1)
    - targets: Stacked (B, 2)
    - ids: Stacked (B,)
    """
    atomic_list, global_list, target_list, id_list = zip(*batch)

    # 1. Concatenate atomic features
    atomic_batch = torch.cat(atomic_list, dim=0)

    # 2. Stack global features
    global_batch = torch.stack(global_list, dim=0)

    # 3. Stack targets and IDs
    target_batch = torch.stack(target_list, dim=0)
    id_batch = torch.stack(id_list, dim=0)

    # 4. Create batch indices
    # For each sample i, create a tensor [i, i, ..., i] of length n_atoms_i
    batch_indices_list = []
    for i, atoms in enumerate(atomic_list):
        n_atoms = atoms.shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

    batch_indices = torch.cat(batch_indices_list, dim=0)

    return {
        "atomic_features": atomic_batch,
        "global_features": global_batch,
        "batch_indices": batch_indices,
        "targets": target_batch,
        "ids": id_batch,
    }
