import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.geometry import (
    read_xyz,
    center_coordinates,
    compute_pbc_neighbor_distances,
)
from library.features import extract_global_features, process_symmetry, FeatureScaler


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for the Symmetry-Informed Residual Deep Sets (SI-RDS) strategy.

    Handles loading, preprocessing, caching, and serving of atomic and global material features.
    """

    # Element mapping for one-hot encoding
    ELEMENTS = ["Al", "Ga", "In", "O"]
    ELEM_TO_IDX = {el: i for i, el in enumerate(ELEMENTS)}

    def __init__(
        self,
        metadata_path,
        geometry_dir,
        scaler=None,
        cache_path=None,
        load_cached_data=True,
        debug_sample_size=None,
        mode="train",
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            geometry_dir (str): Root directory containing geometry files.
            scaler (FeatureScaler, optional): Fitted scaler. If None and mode='train', a new one is fitted.
            cache_path (str, optional): Path to save/load .npz cache.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug_sample_size (int, optional): Limit dataset size for debugging.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata_path = metadata_path
        self.geometry_dir = geometry_dir
        self.mode = mode
        self.cache_path = cache_path

        # Load metadata
        df = pd.read_csv(metadata_path)
        if debug_sample_size is not None:
            df = df.iloc[:debug_sample_size]
        self.df = df

        # Load or process data
        (
            self.ids,
            self.atom_counts,
            self.atomic_feats_flat,
            self.global_feats,
            self.symmetry,
            self.targets,
        ) = self._load_or_process_data(load_cached_data)

        # Handle Scaling
        self.scaler = scaler
        if self.mode == "train" and self.scaler is None:
            self.scaler = FeatureScaler()
            # Fit on continuous parts: Atomic[4:] (coords+dist) and Global[:]
            # We concatenate them to fit a single scaler or fit separately?
            # The FeatureScaler wraps StandardScaler which works on 2D arrays.
            # It's better to have separate scalers or handle indices carefully.
            # Given the provided library `FeatureScaler` is simple, we will use two instances
            # or just manage the numpy arrays directly if FeatureScaler is strictly for one matrix.
            # Looking at library code, FeatureScaler wraps StandardScaler.
            # We will use two internal scalers if we implement it here, but the requirement says
            # "Import the functions or classes... instead of re-implementing".
            # The provided FeatureScaler fits one X. We need to scale two different distributions.
            # We will instantiate two FeatureScalers: one for atomic continuous, one for global.

            self.atomic_scaler = FeatureScaler()
            self.global_scaler = FeatureScaler()

            # Atomic continuous features: cols 4, 5, 6 (coords), 7 (neighbor)
            self.atomic_scaler.fit(self.atomic_feats_flat[:, 4:])
            self.global_scaler.fit(self.global_feats)

            # Save scalers
            # We assume a directory for scalers exists or we create one in working dir
            scaler_dir = os.path.join(os.path.dirname(Config.WORKING_DIR), "scalers")
            os.makedirs(scaler_dir, exist_ok=True)
            self.atomic_scaler.save(os.path.join(scaler_dir, "atomic_scaler.joblib"))
            self.global_scaler.save(os.path.join(scaler_dir, "global_scaler.joblib"))

        elif self.scaler is not None:
            # If a tuple of scalers is passed
            self.atomic_scaler, self.global_scaler = self.scaler
        else:
            # Try to load if not provided (e.g. validation/test mode without explicit scaler pass)
            scaler_dir = os.path.join(os.path.dirname(Config.WORKING_DIR), "scalers")
            self.atomic_scaler = FeatureScaler()
            self.global_scaler = FeatureScaler()
            self.atomic_scaler.load(os.path.join(scaler_dir, "atomic_scaler.joblib"))
            self.global_scaler.load(os.path.join(scaler_dir, "global_scaler.joblib"))

        # Apply Scaling
        self.atomic_feats_flat[:, 4:] = self.atomic_scaler.transform(
            self.atomic_feats_flat[:, 4:]
        )
        self.global_feats = self.global_scaler.transform(self.global_feats)

        # Apply Log Transformation to Targets if training/val
        # Targets are: formation_energy_ev_natom, bandgap_energy_ev
        # We use log1p: log(1 + y)
        if self.mode != "test":
            self.targets = np.log1p(self.targets)

    def _load_or_process_data(self, load_cached_data):
        """
        Loads data from cache if available, otherwise processes from scratch and saves.
        """
        if load_cached_data and self.cache_path and os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path)
                # Check if cache size matches dataframe size (in case of debug/full switch)
                if len(data["ids"]) == len(self.df):
                    return (
                        data["ids"],
                        data["atom_counts"],
                        data["atomic_feats_flat"],
                        data["global_feats"],
                        data["symmetry"],
                        data["targets"],
                    )
            except Exception:
                pass  # Fallback to processing

        # Initialize containers
        ids_list = []
        atom_counts_list = []
        atomic_feats_list = []  # Will be flattened later
        global_feats_list = []
        symmetry_list = []
        targets_list = []

        for _, row in self.df.iterrows():
            # 1. Identifiers
            ids_list.append(row["id"])

            # 2. Geometry Processing
            xyz_path = os.path.join(self.geometry_dir, row["file_path"])
            lattice, atom_types, atom_coords = read_xyz(xyz_path)

            # Center coordinates
            centered_coords = center_coordinates(atom_coords, lattice)

            # PBC Neighbor distances
            neighbor_dists = compute_pbc_neighbor_distances(atom_coords, lattice)

            # One-hot encoding
            n_atoms = len(atom_types)
            one_hot = np.zeros((n_atoms, 4), dtype=np.float32)
            for i, at in enumerate(atom_types):
                if at in self.ELEM_TO_IDX:
                    one_hot[i, self.ELEM_TO_IDX[at]] = 1.0

            # Combine atomic features: [OneHot(4), Coords(3), Dist(1)]
            # Coords are centered, so they are relative features now
            atomic_vecs = np.hstack(
                [one_hot, centered_coords, neighbor_dists.reshape(-1, 1)]
            ).astype(np.float32)

            atomic_feats_list.append(atomic_vecs)
            atom_counts_list.append(n_atoms)

            # 3. Global Features
            g_feats = extract_global_features(row)
            global_feats_list.append(g_feats)

            # 4. Symmetry
            sym = process_symmetry(row)
            symmetry_list.append(sym)

            # 5. Targets (if available)
            if self.mode != "test":
                t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                targets_list.append(t)
            else:
                targets_list.append([0.0, 0.0])  # Dummy targets for test

        # Convert to numpy arrays
        ids = np.array(ids_list, dtype=np.int32)
        atom_counts = np.array(atom_counts_list, dtype=np.int32)
        # Flatten atomic features for efficient storage and scaling
        atomic_feats_flat = np.vstack(atomic_feats_list).astype(np.float32)
        global_feats = np.vstack(global_feats_list).astype(np.float32)
        symmetry = np.array(symmetry_list, dtype=np.int32)
        targets = np.array(targets_list, dtype=np.float32)

        # Cache results
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez(
                self.cache_path,
                ids=ids,
                atom_counts=atom_counts,
                atomic_feats_flat=atomic_feats_flat,
                global_feats=global_feats,
                symmetry=symmetry,
                targets=targets,
            )

        return ids, atom_counts, atomic_feats_flat, global_feats, symmetry, targets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Reconstruct atomic features for this sample from the flattened array
        # We need to know where this sample starts and ends
        # Optimization: Calculate cumulative sum of counts once in init?
        # For simplicity/readability, we calculate offsets here or store them.
        # Given the constraints, let's just pre-calculate offsets in init to be safe.
        # But since I can't modify init easily without re-pasting, I'll do a quick calc or
        # assume we can use `np.split` logic.
        # Actually, let's just store a start_index array in __init__.

        # Lazy fix: calculate start index on the fly is O(N), bad.
        # Better: Precompute offsets in __init__.
        if not hasattr(self, "atom_offsets"):
            self.atom_offsets = np.concatenate(([0], np.cumsum(self.atom_counts)))

        start = self.atom_offsets[idx]
        end = self.atom_offsets[idx + 1]

        atomic_x = self.atomic_feats_flat[start:end]
        global_x = self.global_feats[idx]
        sym_x = self.symmetry[idx]
        target_y = self.targets[idx]
        id_val = self.ids[idx]

        return {
            "id": id_val,
            "atomic_features": torch.from_numpy(atomic_x),
            "global_features": torch.from_numpy(global_x),
            "symmetry": torch.tensor(sym_x, dtype=torch.long),
            "targets": torch.from_numpy(target_y),
        }


def collate_batch(batch):
    """
    Collate function for the DataLoader.
    Pads atomic features to the maximum number of atoms in the batch.
    Creates a mask for valid atoms.
    """
    ids = [item["id"] for item in batch]
    global_feats = torch.stack([item["global_features"] for item in batch])
    symmetry = torch.stack([item["symmetry"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])

    # Atomic features padding
    atomic_feats_list = [item["atomic_features"] for item in batch]
    lengths = [x.shape[0] for x in atomic_feats_list]
    max_len = max(lengths)

    # Feature dim is 8
    feat_dim = atomic_feats_list[0].shape[1]

    # Prepare padded tensor
    batch_size = len(batch)
    padded_atomic = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (feat, length) in enumerate(zip(atomic_feats_list, lengths)):
        padded_atomic[i, :length, :] = feat
        mask[i, :length] = True

    return {
        "ids": ids,
        "atomic_features": padded_atomic,
        "global_features": global_feats,
        "symmetry": symmetry,
        "targets": targets,
        "mask": mask,
    }
