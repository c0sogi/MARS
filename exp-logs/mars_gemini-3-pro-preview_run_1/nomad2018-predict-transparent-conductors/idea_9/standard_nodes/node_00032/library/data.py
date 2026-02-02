import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ase.io
from library.config import Config
from library.utils import compute_pbc_distance


class Scaler:
    """
    Handles standardization (z-score scaling) of continuous features.
    Fits on training data and transforms inputs.
    """

    def __init__(self):
        self.atom_mean = None
        self.atom_std = None
        self.global_mean = None
        self.global_std = None
        self.fitted = False

    def fit(self, atom_continuous, global_features):
        """
        Compute mean and std for atomic continuous features (coords + nn_dist)
        and global features.

        Args:
            atom_continuous (np.ndarray): Shape (N_total_atoms, 4) -> [x, y, z, nn_dist]
            global_features (np.ndarray): Shape (N_samples, 12)
        """
        self.atom_mean = np.mean(atom_continuous, axis=0)
        self.atom_std = np.std(atom_continuous, axis=0)
        # Prevent division by zero for constant features (if any)
        self.atom_std[self.atom_std < 1e-6] = 1.0

        self.global_mean = np.mean(global_features, axis=0)
        self.global_std = np.std(global_features, axis=0)
        self.global_std[self.global_std < 1e-6] = 1.0

        self.fitted = True

    def transform_atomic(self, atom_continuous):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")
        return (atom_continuous - self.atom_mean) / self.atom_std

    def transform_global(self, global_features):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")
        return (global_features - self.global_mean) / self.global_std

    def state_dict(self):
        return {
            "atom_mean": self.atom_mean,
            "atom_std": self.atom_std,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "fitted": self.fitted,
        }

    def load_state_dict(self, state_dict):
        self.atom_mean = state_dict["atom_mean"]
        self.atom_std = state_dict["atom_std"]
        self.global_mean = state_dict["global_mean"]
        self.global_std = state_dict["global_std"]
        self.fitted = state_dict["fitted"]


def process_data(metadata_path, cache_path, load_cached_data=True):
    """
    Parses geometry files, computes features, and caches the result.

    Args:
        metadata_path (str): Path to the csv file (train/val/test).
        cache_path (str): Path to save/load the .npz cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Lists to collect data
    all_atom_types = []
    all_atom_coords = []
    all_atom_nn_dists = []
    all_batch_indices = []  # Maps atom to sample index

    all_global_feats = []
    all_spacegroups = []
    all_targets = []
    all_ids = []

    # For reconstruction in Dataset
    sample_atom_counts = []

    # Pre-compute atom type mapping
    type_map = {sym: i for i, sym in enumerate(Config.ATOM_TYPES)}

    for idx, row in df.iterrows():
        # --- Geometry Processing ---
        # Construct full path to geometry file
        # row['file_path'] is relative to input dir, e.g. "train/1/geometry.xyz"
        xyz_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read atoms object
        atoms = ase.io.read(xyz_path, format="aims")
        positions = atoms.get_positions()
        lattice = atoms.get_cell()[:]

        # 1. Center coordinates
        centroid = np.mean(positions, axis=0)
        centered_pos = positions - centroid

        # 2. PBC Nearest Neighbor Distance
        # Using the utility function provided in library.utils
        nn_dists = compute_pbc_distance(positions, lattice)

        # 3. Atom Types
        symbols = atoms.get_chemical_symbols()
        type_indices = [type_map[s] for s in symbols]

        # Collect Atomic Data
        n_atoms = len(symbols)
        all_atom_types.append(np.array(type_indices, dtype=np.int64))
        all_atom_coords.append(centered_pos.astype(np.float32))
        all_atom_nn_dists.append(nn_dists.astype(np.float32))
        all_batch_indices.append(np.full(n_atoms, idx, dtype=np.int64))
        sample_atom_counts.append(n_atoms)

        # --- Global Features ---
        # Lattice lengths (a, b, c) and angles (alpha, beta, gamma)
        # Stoichiometry (Al, Ga, In) - O is implicit/dependent
        # Total atoms
        # Volume
        # Density

        # Lattice parameters from metadata
        lat_params = [
            row["lattice_vector_1_ang"],
            row["lattice_vector_2_ang"],
            row["lattice_vector_3_ang"],
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

        # Derived physics
        # Volume calculation from lattice parameters
        a, b, c = lat_params[0], lat_params[1], lat_params[2]
        alpha = np.radians(lat_params[3])
        beta = np.radians(lat_params[4])
        gamma = np.radians(lat_params[5])

        vol_term = (
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        volume = a * b * c * np.sqrt(max(0, vol_term))

        total_atoms = row["number_of_total_atoms"]
        density = total_atoms / (volume + 1e-6)

        # Assemble global vector (Dim: 6 + 3 + 1 + 1 + 1 = 12)
        global_vec = lat_params + stoich + [total_atoms, volume, density]
        all_global_feats.append(global_vec)

        # --- Other Metadata ---
        all_spacegroups.append(row["spacegroup"])
        all_ids.append(row["id"])

        # Targets (handle missing for test set)
        if "formation_energy_ev_natom" in row:
            # Apply log1p transformation: log(1 + y)
            # Targets must be positive for log1p. Energy is usually positive or small negative.
            # Task description implies energies. Formation energy can be negative, but metric is RMSLE.
            # RMSLE implies targets are non-negative. We assume provided targets are valid for RMSLE.
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            all_targets.append([t1, t2])
        else:
            all_targets.append([np.nan, np.nan])

    # Concatenate all data
    data_dict = {
        "atom_types": np.concatenate(all_atom_types),
        "atom_coords": np.concatenate(all_atom_coords),
        "atom_nn_dists": np.concatenate(all_atom_nn_dists),
        "batch_map": np.concatenate(
            all_batch_indices
        ),  # Maps atom index -> sample index
        "sample_atom_counts": np.array(sample_atom_counts, dtype=np.int64),
        "global_features": np.array(all_global_feats, dtype=np.float32),
        "spacegroups": np.array(all_spacegroups, dtype=np.int64),
        "targets": np.array(all_targets, dtype=np.float32),
        "ids": np.array(all_ids, dtype=np.int64),
    }

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material energy prediction.
    Handles on-the-fly scaling and sample retrieval.
    """

    def __init__(self, data_dict, scaler=None, phase="train"):
        """
        Args:
            data_dict (dict): Dictionary of numpy arrays from process_data.
            scaler (Scaler): Fitted scaler instance. If None and phase='train', will fit a new one.
            phase (str): 'train', 'val', or 'test'.
        """
        self.data = data_dict
        self.phase = phase
        self.num_samples = len(self.data["ids"])

        # Pre-calculate start indices for atoms to allow O(1) access per sample
        # sample_atom_counts gives length of each sample.
        # cumsum gives end indices.
        counts = self.data["sample_atom_counts"]
        self.atom_slices = np.concatenate(([0], np.cumsum(counts)))

        # Handle Scaler
        if scaler is None:
            if phase == "train":
                self.scaler = Scaler()
                # Prepare data for fitting
                # Combine coords and nn_dist for continuous atomic features
                atom_cont = np.hstack(
                    [self.data["atom_coords"], self.data["atom_nn_dists"][:, None]]
                )
                self.scaler.fit(atom_cont, self.data["global_features"])
            else:
                raise ValueError("Scaler must be provided for validation/test sets.")
        else:
            self.scaler = scaler

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Get Atomic Data
        start_atom = self.atom_slices[idx]
        end_atom = self.atom_slices[idx + 1]

        # Raw features
        types = self.data["atom_types"][start_atom:end_atom]  # (N,)
        coords = self.data["atom_coords"][start_atom:end_atom]  # (N, 3)
        dists = self.data["atom_nn_dists"][start_atom:end_atom]  # (N,)

        # Scale continuous atomic features
        raw_cont = np.hstack([coords, dists[:, None]])  # (N, 4)
        scaled_cont = self.scaler.transform_atomic(raw_cont)

        # One-hot encode types
        # Config.NUM_ATOM_TYPES = 4
        one_hot = np.zeros((len(types), Config.NUM_ATOM_TYPES), dtype=np.float32)
        one_hot[np.arange(len(types)), types] = 1.0

        # Combine to form full atomic feature vector
        # [OneHot(4), Scaled_Coords(3), Scaled_Dist(1)] -> Dim 8
        atom_features = np.hstack([one_hot, scaled_cont]).astype(np.float32)

        # 2. Get Global Data
        raw_global = self.data["global_features"][idx]  # (12,)
        scaled_global = self.scaler.transform_global(raw_global[None, :])[0].astype(
            np.float32
        )

        # 3. Other Metadata
        spacegroup = self.data["spacegroups"][idx]
        sample_id = self.data["ids"][idx]
        target = self.data["targets"][idx]

        return {
            "atom_features": torch.from_numpy(atom_features),
            "global_features": torch.from_numpy(scaled_global),
            "spacegroup": torch.tensor(spacegroup, dtype=torch.long),
            "target": torch.from_numpy(target),
            "id": torch.tensor(sample_id, dtype=torch.long),
        }


class CollateFn:
    """
    Custom collate function to batch variable-sized point clouds.
    Creates a packed batch for atomic features and standard batches for global features.
    """

    def __call__(self, batch):
        # batch is a list of dicts from __getitem__

        # 1. Atomic Features
        # Concatenate all atoms from all samples along dim 0
        atom_features_list = [sample["atom_features"] for sample in batch]
        packed_atom_features = torch.cat(atom_features_list, dim=0)

        # Create batch index vector (0,0,0, 1,1, 2,2,2, ...) for scatter operations
        batch_indices_list = [
            torch.full((sample["atom_features"].shape[0],), i, dtype=torch.long)
            for i, sample in enumerate(batch)
        ]
        batch_indices = torch.cat(batch_indices_list, dim=0)

        # 2. Global Features & Others
        global_features = torch.stack([sample["global_features"] for sample in batch])
        spacegroups = torch.stack([sample["spacegroup"] for sample in batch])
        targets = torch.stack([sample["target"] for sample in batch])
        ids = torch.stack([sample["id"] for sample in batch])

        return {
            "atom_features": packed_atom_features,  # (Sum_N, 8)
            "batch_indices": batch_indices,  # (Sum_N,)
            "global_features": global_features,  # (B, 12)
            "spacegroups": spacegroups,  # (B,)
            "targets": targets,  # (B, 2)
            "ids": ids,  # (B,)
        }


def get_datasets(load_cached_data=True):
    """
    Factory function to prepare Train, Val, and Test datasets.
    Handles caching and scaler fitting.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Process/Load Data
    train_data = process_data(
        Config.TRAIN_META_PATH, Config.TRAIN_DATA_CACHE, load_cached_data
    )
    val_data = process_data(
        Config.VAL_META_PATH, Config.VAL_DATA_CACHE, load_cached_data
    )
    test_data = process_data(
        Config.TEST_META_PATH, Config.TEST_DATA_CACHE, load_cached_data
    )

    # 2. Create Datasets
    # Train dataset fits the scaler
    train_dataset = MaterialDataset(train_data, scaler=None, phase="train")

    # Val and Test reuse the fitted scaler
    scaler = train_dataset.scaler
    val_dataset = MaterialDataset(val_data, scaler=scaler, phase="val")
    test_dataset = MaterialDataset(test_data, scaler=scaler, phase="test")

    return train_dataset, val_dataset, test_dataset
