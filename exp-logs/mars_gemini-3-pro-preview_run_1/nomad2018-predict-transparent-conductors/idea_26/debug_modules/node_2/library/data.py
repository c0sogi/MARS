import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library import config, features


class CrystalDataset(Dataset):
    def __init__(self, atomic_features, batch_indices, global_features, targets, ids):
        """
        PyTorch Dataset for crystal structures.

        Args:
            atomic_features (np.ndarray): (Total_Atoms, 12) array of atomic features.
            batch_indices (np.ndarray): (Total_Atoms,) array mapping each atom to its crystal index (0 to N-1).
            global_features (np.ndarray): (N_crystals, 12) array of global crystal features.
            targets (np.ndarray): (N_crystals, 2) array of target values.
            ids (np.ndarray): (N_crystals,) array of crystal IDs.
        """
        self.atomic_features = torch.tensor(atomic_features, dtype=torch.float32)
        self.batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        self.global_features = torch.tensor(global_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.long)

        # Pre-calculate start indices and counts for each crystal to allow O(1) slicing in __getitem__
        # Assumes batch_indices are sorted and contiguous (0, 0, ..., 1, 1, ..., N-1, N-1)
        # This is guaranteed by the sequential processing in features.py
        unique_ids, counts = torch.unique(self.batch_indices, return_counts=True)

        # Ensure we have counts for all crystals, even if some have 0 atoms (unlikely but safe)
        self.num_crystals = len(global_features)

        # atom_counts[i] = number of atoms in crystal i
        self.atom_counts = counts

        # atom_starts[i] = starting index in atomic_features for crystal i
        # We prepend 0 to the cumulative sum to get start indices
        self.atom_starts = torch.cat(
            [torch.tensor([0]), torch.cumsum(counts, dim=0)[:-1]]
        )

    def __len__(self):
        return self.num_crystals

    def __getitem__(self, idx):
        """
        Returns data for a single crystal.
        """
        start = self.atom_starts[idx]
        count = self.atom_counts[idx]
        end = start + count

        # Slice atomic features corresponding to this crystal
        atoms = self.atomic_features[start:end]

        # Get global features and target for this crystal
        glob = self.global_features[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return atoms, glob, target, sample_id


def collate_crystals(batch):
    """
    Collate function to batch variable-sized crystal graphs.

    Args:
        batch: List of tuples (atoms, glob, target, sample_id)

    Returns:
        batched_atoms: (Total_Batch_Atoms, Atom_Feat_Dim)
        batch_vec: (Total_Batch_Atoms,) mapping atoms to their batch element index
        batched_global: (Batch_Size, Global_Feat_Dim)
        batched_targets: (Batch_Size, Target_Dim)
        batched_ids: (Batch_Size,)
    """
    atoms_list, glob_list, target_list, id_list = zip(*batch)

    # 1. Concatenate all atomic features into one large tensor
    batched_atoms = torch.cat(atoms_list, dim=0)

    # 2. Create a batch vector that maps each atom to its crystal index in the batch
    # e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2, ...]
    batch_vec_list = []
    for i, atoms in enumerate(atoms_list):
        n_atoms = atoms.shape[0]
        batch_vec_list.append(torch.full((n_atoms,), i, dtype=torch.long))
    batch_vec = torch.cat(batch_vec_list, dim=0)

    # 3. Stack global features, targets, and IDs (fixed size per crystal)
    batched_global = torch.stack(glob_list, dim=0)
    batched_targets = torch.stack(target_list, dim=0)
    batched_ids = torch.stack(id_list, dim=0)

    return batched_atoms, batch_vec, batched_global, batched_targets, batched_ids


class DataProcessor:
    def __init__(self):
        self.atom_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.is_fitted = False

        # Indices for continuous columns in atomic features that require scaling
        # 0-3: One-hot encoding (Al, Ga, In, O) -> Do NOT scale
        # 4-6: Centered Coordinates (x, y, z) -> Scale
        # 7-10: Inverse Proximity -> Scale
        # 11: Local Packing Density -> Scale
        self.atom_cont_indices = [4, 5, 6, 7, 8, 9, 10, 11]

    def _fit_scalers(self, atomic_feats, global_feats):
        """Fits scalers on training data and saves them."""
        # Fit atomic scaler only on continuous features
        self.atom_scaler.fit(atomic_feats[:, self.atom_cont_indices])

        # Fit global scaler on all global features (all are continuous)
        self.global_scaler.fit(global_feats)

        self.is_fitted = True

        # Save scaler parameters to npz (avoiding pickle)
        np.savez(
            config.SCALERS_CACHE,
            atom_mean=self.atom_scaler.mean_,
            atom_scale=self.atom_scaler.scale_,
            global_mean=self.global_scaler.mean_,
            global_scale=self.global_scaler.scale_,
        )

    def _load_scalers(self):
        """Loads scaler parameters from cache if available."""
        if os.path.exists(config.SCALERS_CACHE):
            try:
                data = np.load(config.SCALERS_CACHE)
                self.atom_scaler.mean_ = data["atom_mean"]
                self.atom_scaler.scale_ = data["atom_scale"]
                self.atom_scaler.var_ = data["atom_scale"] ** 2  # Reconstruct variance

                self.global_scaler.mean_ = data["global_mean"]
                self.global_scaler.scale_ = data["global_scale"]
                self.global_scaler.var_ = data["global_scale"] ** 2

                self.is_fitted = True
                return True
            except Exception as e:
                print(f"Error loading scalers: {e}")
                return False
        return False

    def _transform(self, atomic_feats, global_feats):
        """Applies standardization to features."""
        if not self.is_fitted:
            raise RuntimeError("Scalers must be fitted before transform.")

        # Transform atomic continuous features
        # Create a copy to avoid modifying original data in place if needed later
        atomic_feats_scaled = atomic_feats.copy()
        atomic_feats_scaled[:, self.atom_cont_indices] = self.atom_scaler.transform(
            atomic_feats[:, self.atom_cont_indices]
        )

        # Transform global features
        global_feats_scaled = self.global_scaler.transform(global_feats)

        return atomic_feats_scaled, global_feats_scaled

    def process_and_get_loaders(self, load_cached_data=True):
        """
        Orchestrates data loading, feature extraction, scaling, and DataLoader creation.

        Args:
            load_cached_data (bool): If True, attempts to load processed features from disk.

        Returns:
            train_loader, val_loader, test_loader
        """
        # 1. Load/Compute Raw Features using the features library
        # This handles the heavy lifting of parsing XYZ files and computing geometric features
        train_data = features.prepare_features(
            config.TRAIN_CSV, config.TRAIN_DATA_CACHE, load_cached_data
        )
        val_data = features.prepare_features(
            config.VAL_CSV, config.VAL_DATA_CACHE, load_cached_data
        )
        test_data = features.prepare_features(
            config.TEST_CSV, config.TEST_DATA_CACHE, load_cached_data
        )

        # 2. Fit or Load Scalers
        # We only fit on training data
        if load_cached_data and self._load_scalers():
            print("Loaded scalers from cache.")
        else:
            print("Fitting scalers on training data...")
            self._fit_scalers(
                train_data["atomic_features"], train_data["global_features"]
            )

        # 3. Apply Scaling
        train_atoms, train_glob = self._transform(
            train_data["atomic_features"], train_data["global_features"]
        )
        val_atoms, val_glob = self._transform(
            val_data["atomic_features"], val_data["global_features"]
        )
        test_atoms, test_glob = self._transform(
            test_data["atomic_features"], test_data["global_features"]
        )

        # 4. Target Transformation
        # Apply log1p(y) to targets to match RMSLE metric optimization
        train_targets = np.log1p(train_data["targets"])
        val_targets = np.log1p(val_data["targets"])
        # Test targets are placeholders, keep as is (or transform, doesn't matter)
        test_targets = test_data["targets"]

        # 5. Create Datasets
        train_dataset = CrystalDataset(
            train_atoms,
            train_data["batch_indices"],
            train_glob,
            train_targets,
            train_data["ids"],
        )
        val_dataset = CrystalDataset(
            val_atoms, val_data["batch_indices"], val_glob, val_targets, val_data["ids"]
        )
        test_dataset = CrystalDataset(
            test_atoms,
            test_data["batch_indices"],
            test_glob,
            test_targets,
            test_data["ids"],
        )

        # 6. Create DataLoaders
        # Use num_workers for parallel data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_crystals,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_crystals,
            num_workers=4,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_crystals,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def inverse_transform_targets(self, targets_log1p):
        """
        Reverses the log1p transformation: exp(y) - 1.
        Used for generating final submission predictions.
        """
        return np.expm1(targets_log1p)
