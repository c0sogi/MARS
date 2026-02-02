import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import (
    parse_xyz,
    compute_pbc_distances,
    calculate_idw_chemical_counts,
)


class MaterialDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, sample_size=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache.
            sample_size (int, optional): If set, limits the dataset size for debugging.
        """
        self.mode = mode
        self.sample_size = sample_size

        # Determine paths based on mode
        if mode == "train":
            self.metadata_path = Config.TRAIN_METADATA
            self.cache_path = Config.TRAIN_DATA_CACHE
        elif mode == "val":
            self.metadata_path = Config.VAL_METADATA
            self.cache_path = Config.VAL_DATA_CACHE
        else:
            self.metadata_path = Config.TEST_METADATA
            self.cache_path = Config.TEST_DATA_CACHE

        # Load or process data
        self.atomic_features = []
        self.global_features = []
        self.targets = []
        self.ids = []

        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        # Check if cache exists
        if (
            load_cached_data
            and os.path.exists(self.cache_path)
            and os.path.exists(Config.SCALERS_CACHE)
        ):
            print(f"Loading cached {self.mode} data from {self.cache_path}...")
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self.atomic_features = list(data["atomic_features"])
                self.global_features = data["global_features"].astype(np.float32)
                self.targets = data["targets"].astype(np.float32)
                self.ids = data["ids"]

                # If debugging, slice data
                if self.sample_size is not None:
                    self.atomic_features = self.atomic_features[: self.sample_size]
                    self.global_features = self.global_features[: self.sample_size]
                    self.targets = self.targets[: self.sample_size]
                    self.ids = self.ids[: self.sample_size]
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {self.mode} data from scratch...")
        df = pd.read_csv(self.metadata_path)

        if self.sample_size is not None:
            df = df.head(self.sample_size)

        # Load scalers if they exist, otherwise we will fit them (only if train)
        atomic_scaler = StandardScaler()
        global_scaler = StandardScaler()

        scalers_exist = os.path.exists(Config.SCALERS_CACHE)

        if self.mode == "train" and not scalers_exist:
            # We will fit later after collecting all data
            pass
        elif scalers_exist:
            # Load scalers params
            scaler_data = np.load(Config.SCALERS_CACHE)
            atomic_scaler.mean_ = scaler_data["atomic_mean"]
            atomic_scaler.scale_ = scaler_data["atomic_scale"]
            global_scaler.mean_ = scaler_data["global_mean"]
            global_scaler.scale_ = scaler_data["global_scale"]
        else:
            if self.mode != "train":
                raise FileNotFoundError(
                    "Scalers not found. Run training set first to generate scalers."
                )

        # Temporary lists for raw features
        raw_atomic_continuous = []  # Coords(3), NN(1), IDW(4) -> 8 dims
        raw_atomic_onehot = []  # Onehot(4)
        raw_global = []  # 12 dims
        raw_targets = []
        raw_ids = []

        # Track indices to reconstruct structure later
        sample_atom_counts = []

        for idx, row in df.iterrows():
            mat_id = row["id"]

            # 1. Global Features from CSV
            lat_lens = [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
            ]
            lat_angs = [
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
            ]

            # Stoichiometry
            stoich = [
                row["percent_atom_al"],
                row["percent_atom_ga"],
                row["percent_atom_in"],
            ]

            # Total atoms
            n_atoms = row["number_of_total_atoms"]

            # Calculate Volume and Density
            alpha, beta, gamma = np.radians(lat_angs)
            a, b, c = lat_lens
            term = (
                1
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
            volume = a * b * c * np.sqrt(max(0, term))
            density = n_atoms / volume

            g_feats = lat_lens + lat_angs + [volume, density] + stoich + [n_atoms]
            raw_global.append(g_feats)

            # Targets: Log(1+y) transformation
            if "formation_energy_ev_natom" in row:
                t1 = np.log1p(row["formation_energy_ev_natom"])
                t2 = np.log1p(row["bandgap_energy_ev"])
                raw_targets.append([t1, t2])
            else:
                raw_targets.append([0.0, 0.0])  # Placeholder

            raw_ids.append(mat_id)

            # 2. Atomic Features from XYZ
            xyz_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            lattice, atom_types, coords = parse_xyz(xyz_path)

            # Centering
            if len(coords) > 0:
                centroid = np.mean(coords, axis=0)
                centered_coords = coords - centroid
            else:
                centered_coords = np.zeros((0, 3), dtype=np.float32)

            # PBC Distances & NN Dist
            # Cite debug_lesson_9: Explicitly Handle Empty Sets
            if len(coords) > 0:
                dists = compute_pbc_distances(coords, lattice)
                # Mask diagonal to ignore self-distance
                np.fill_diagonal(dists, np.inf)
                nn_dist = np.min(dists, axis=1).reshape(-1, 1)
            else:
                # If empty, return empty array of shape (0, 1)
                nn_dist = np.zeros((0, 1), dtype=np.float32)

            # IDW-CC
            idw_cc = calculate_idw_chemical_counts(
                coords, atom_types, lattice, k=Config.K_NEIGHBORS
            )

            # One-hot encoding
            one_hot = np.zeros(
                (len(atom_types), Config.NUM_ATOM_TYPES), dtype=np.float32
            )
            for i, at in enumerate(atom_types):
                if at in Config.ATOM_MAP:
                    one_hot[i, Config.ATOM_MAP[at]] = 1.0

            # Collect continuous atomic features for scaling
            # Concatenate: Coords (3) + NN Dist (1) + IDW-CC (4) = 8 dims
            cont_feats = np.hstack([centered_coords, nn_dist, idw_cc])

            raw_atomic_continuous.append(cont_feats)
            raw_atomic_onehot.append(one_hot)
            sample_atom_counts.append(len(atom_types))

        # Convert to numpy
        raw_global = np.array(raw_global, dtype=np.float32)
        all_atomic_cont = np.vstack(raw_atomic_continuous).astype(np.float32)

        # Fit Scalers if training
        if self.mode == "train" and not scalers_exist:
            print("Fitting scalers on training data...")
            atomic_scaler.fit(all_atomic_cont)
            global_scaler.fit(raw_global)

            # Save scalers
            np.savez(
                Config.SCALERS_CACHE,
                atomic_mean=atomic_scaler.mean_,
                atomic_scale=atomic_scaler.scale_,
                global_mean=global_scaler.mean_,
                global_scale=global_scaler.scale_,
            )

        # Transform
        scaled_global = global_scaler.transform(raw_global)
        scaled_atomic_cont = atomic_scaler.transform(all_atomic_cont)

        # Reconstruct atomic features per sample
        self.atomic_features = []
        cursor = 0
        for i, count in enumerate(sample_atom_counts):
            cont_part = scaled_atomic_cont[cursor : cursor + count]
            onehot_part = raw_atomic_onehot[i]
            # Combine: One-hot (4) + Scaled Continuous (8) -> Total 12
            full_atomic = np.hstack([onehot_part, cont_part])
            self.atomic_features.append(full_atomic)
            cursor += count

        self.global_features = scaled_global
        self.targets = np.array(raw_targets, dtype=np.float32)
        self.ids = np.array(raw_ids, dtype=np.int32)

        # Save to cache
        print(f"Saving processed {self.mode} data to {self.cache_path}...")
        # atomic_features is a list of arrays of different lengths (object array)
        np.savez(
            self.cache_path,
            atomic_features=np.array(self.atomic_features, dtype=object),
            global_features=self.global_features,
            targets=self.targets,
            ids=self.ids,
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.atomic_features[idx], dtype=torch.float32),
            torch.tensor(self.global_features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.ids[idx], dtype=torch.int32),
        )


def collate_batch(batch):
    """
    Collate function for batching variable-size atomic structures.

    Args:
        batch: List of tuples (atomic_feats, global_feats, targets, ids)

    Returns:
        dict: containing:
            - atomic_features: (B, N_max, F_atomic) padded tensor
            - atomic_mask: (B, N_max) boolean tensor (True for valid atoms)
            - global_features: (B, F_global) tensor
            - targets: (B, 2) tensor
            - ids: (B,) tensor
    """
    atomic_feats_list, global_feats_list, targets_list, ids_list = zip(*batch)

    batch_size = len(atomic_feats_list)
    # Handle case where all graphs in batch are empty (unlikely but possible)
    max_atoms = max([f.shape[0] for f in atomic_feats_list])
    if max_atoms == 0:
        max_atoms = 1  # Prevent zero-size tensor creation issues

    feat_dim = (
        atomic_feats_list[0].shape[1]
        if atomic_feats_list[0].shape[0] > 0
        else Config.ATOMIC_FEATURE_DIM
    )

    # Pad atomic features
    padded_atomic = torch.zeros((batch_size, max_atoms, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)

    for i, feats in enumerate(atomic_feats_list):
        n_atoms = feats.shape[0]
        if n_atoms > 0:
            padded_atomic[i, :n_atoms, :] = feats
            mask[i, :n_atoms] = True

    return {
        "atomic_features": padded_atomic,
        "atomic_mask": mask,
        "global_features": torch.stack(global_feats_list),
        "targets": torch.stack(targets_list),
        "ids": torch.stack(ids_list),
    }
