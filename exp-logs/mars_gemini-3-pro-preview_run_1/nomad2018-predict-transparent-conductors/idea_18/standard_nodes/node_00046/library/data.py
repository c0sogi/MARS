import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import (
    parse_xyz,
    calculate_cell_volume,
    calculate_atomic_density,
    get_chemical_neighbor_distances,
)


class SimpleScaler:
    """
    A simple standard scaler that can be saved/loaded using numpy,
    avoiding pickle to comply with requirements.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Prevent division by zero
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            # If not fitted, return data as is (or raise error)
            # For robustness in edge cases, we might raise, but here we assume correct usage.
            raise ValueError("Scaler has not been fitted.")
        return (X - self.mean) / self.scale

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.scale = data["scale"]
        return self


def get_one_hot(atom_types, categories):
    """
    Creates one-hot encoding for atom types.
    """
    mapping = {atype: i for i, atype in enumerate(categories)}
    one_hot = np.zeros((len(atom_types), len(categories)), dtype=np.float32)
    for i, atype in enumerate(atom_types):
        if atype in mapping:
            one_hot[i, mapping[atype]] = 1.0
    return one_hot


def process_data(metadata_path, input_dir, atom_categories):
    """
    Reads metadata and processes raw files into features.
    Returns a dictionary of numpy arrays.
    """
    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    all_global_features = []
    all_targets = []
    all_ids = []
    crystal_sizes = []

    for idx, row in df.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])

        # 1. Parse Geometry
        data = parse_xyz(file_path)
        coords = data["coords"]
        atom_types = data["atom_types"]
        lattice_vectors = data["lattice_vectors"]

        num_atoms = len(atom_types)
        crystal_sizes.append(num_atoms)
        all_ids.append(row["id"])

        # 2. Atomic Features
        # a. One-hot encoding
        one_hot = get_one_hot(atom_types, atom_categories)

        # b. Centered Coords
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # c. Chemical Neighbors (Chemically-Resolved Nearest Neighbor Distances)
        chem_dists = get_chemical_neighbor_distances(
            coords, atom_types, lattice_vectors, atom_categories
        )

        # Combine Atomic Features
        # Structure: [OneHot(4), Coords(3), ChemDists(4)] -> 11 dims
        atomic_feat = np.hstack([one_hot, centered_coords, chem_dists])
        all_atomic_features.append(atomic_feat)

        # 3. Global Features
        # Lattice lengths and angles
        lv1 = row["lattice_vector_1_ang"]
        lv2 = row["lattice_vector_2_ang"]
        lv3 = row["lattice_vector_3_ang"]
        alpha = row["lattice_angle_alpha_degree"]
        beta = row["lattice_angle_beta_degree"]
        gamma = row["lattice_angle_gamma_degree"]

        # Derived physical properties
        vol = calculate_cell_volume(lattice_vectors)
        density = calculate_atomic_density(num_atoms, vol)

        # Stoichiometry (Al, Ga, In)
        stoich = [
            row.get("percent_atom_al", 0.0),
            row.get("percent_atom_ga", 0.0),
            row.get("percent_atom_in", 0.0),
        ]

        # Combine Global Features
        # Structure: [L1, L2, L3, A, B, G, Vol, Dens, S_Al, S_Ga, S_In, N_atoms] -> 12 dims
        global_feat = np.array(
            [
                lv1,
                lv2,
                lv3,
                alpha,
                beta,
                gamma,
                vol,
                density,
                stoich[0],
                stoich[1],
                stoich[2],
                float(num_atoms),
            ],
            dtype=np.float32,
        )
        all_global_features.append(global_feat)

        # 4. Targets
        if "formation_energy_ev_natom" in row:
            t1 = row["formation_energy_ev_natom"]
            t2 = row["bandgap_energy_ev"]
            # Apply log(1+y) transformation
            all_targets.append([np.log1p(t1), np.log1p(t2)])
        else:
            # Placeholder for test set
            all_targets.append([0.0, 0.0])

    # Flatten atomic features for storage efficiency and simplicity
    flat_atomic = np.vstack(all_atomic_features).astype(np.float32)
    stacked_global = np.vstack(all_global_features).astype(np.float32)
    stacked_targets = np.vstack(all_targets).astype(np.float32)
    ids_array = np.array(all_ids, dtype=np.int32)
    sizes_array = np.array(crystal_sizes, dtype=np.int32)

    return {
        "atomic_features": flat_atomic,
        "global_features": stacked_global,
        "targets": stacked_targets,
        "ids": ids_array,
        "crystal_sizes": sizes_array,
    }


class CrystalDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_path,
        scalers_path=None,
        split="train",
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path: Path to train.csv, val.csv, or test.csv
            cache_path: Path to .npz file for cached data
            scalers_path: Path to .npz file for scalers (saved if train, loaded if val/test)
            split: 'train', 'val', or 'test'
            load_cached_data: Whether to try loading from cache
        """
        self.split = split
        self.scalers_path = scalers_path

        # 1. Load or Compute Data
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}...")
            data = np.load(cache_path)
            self.atomic_features = data["atomic_features"]
            self.global_features = data["global_features"]
            self.targets = data["targets"]
            self.ids = data["ids"]
            self.crystal_sizes = data["crystal_sizes"]
        else:
            print(f"Processing {split} data from scratch...")
            data = process_data(metadata_path, Config.INPUT_DIR, Config.ATOM_TYPES)

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(
                cache_path,
                atomic_features=data["atomic_features"],
                global_features=data["global_features"],
                targets=data["targets"],
                ids=data["ids"],
                crystal_sizes=data["crystal_sizes"],
            )

            self.atomic_features = data["atomic_features"]
            self.global_features = data["global_features"]
            self.targets = data["targets"]
            self.ids = data["ids"]
            self.crystal_sizes = data["crystal_sizes"]

        # 2. Handle Scaling
        # Atomic features: indices 4:11 are continuous (coords + distances)
        # Global features: all 12 are continuous

        self.atomic_scaler = SimpleScaler()
        self.global_scaler = SimpleScaler()

        if split == "train":
            # Fit scalers on training data
            # Select continuous atomic cols (skip one-hot)
            atomic_cont = self.atomic_features[:, 4:]
            self.atomic_scaler.fit(atomic_cont)
            self.global_scaler.fit(self.global_features)

            # Save scalers
            if scalers_path:
                os.makedirs(os.path.dirname(scalers_path), exist_ok=True)
                np.savez(
                    scalers_path,
                    atomic_mean=self.atomic_scaler.mean,
                    atomic_scale=self.atomic_scaler.scale,
                    global_mean=self.global_scaler.mean,
                    global_scale=self.global_scaler.scale,
                )
        else:
            # Load scalers for val/test
            if scalers_path and os.path.exists(scalers_path):
                sdata = np.load(scalers_path)
                self.atomic_scaler.mean = sdata["atomic_mean"]
                self.atomic_scaler.scale = sdata["atomic_scale"]
                self.global_scaler.mean = sdata["global_mean"]
                self.global_scaler.scale = sdata["global_scale"]
            else:
                print(
                    "Warning: Scalers not found for validation/test set. Data will be unscaled."
                )

        # 3. Apply Scaling (Transform)
        # Only transform if scalers are fitted
        if self.atomic_scaler.mean is not None:
            self.atomic_features[:, 4:] = self.atomic_scaler.transform(
                self.atomic_features[:, 4:]
            )

        if self.global_scaler.mean is not None:
            self.global_features = self.global_scaler.transform(self.global_features)

        # Pre-calculate cumulative sizes for indexing to reconstruct batches
        self.cumulative_sizes = np.concatenate(([0], np.cumsum(self.crystal_sizes)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Extract atomic features for this crystal using cumulative sizes
        start = self.cumulative_sizes[idx]
        end = self.cumulative_sizes[idx + 1]

        atom_feats = self.atomic_features[start:end]
        glob_feats = self.global_features[idx]
        target = self.targets[idx]
        crystal_id = self.ids[idx]

        return {
            "atomic_features": torch.tensor(atom_feats, dtype=torch.float32),
            "global_features": torch.tensor(glob_feats, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "id": int(crystal_id),
        }


def collate_crystals(batch):
    """
    Collates a list of dictionaries into a batch.
    Atomic features are concatenated (flattened).
    Batch indices are generated to map atoms to crystals.
    Global features and targets are stacked.
    """
    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []
    batch_indices_list = []

    for i, item in enumerate(batch):
        atomic_features_list.append(item["atomic_features"])
        global_features_list.append(item["global_features"])
        targets_list.append(item["target"])
        ids_list.append(item["id"])

        # Create batch index for this crystal's atoms
        n_atoms = item["atomic_features"].shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

    # Concatenate atomic features and batch indices
    batch_atomic = torch.cat(atomic_features_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # Stack global features and targets
    batch_global = torch.stack(global_features_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)

    return {
        "atomic_features": batch_atomic,
        "batch_index": batch_indices,
        "global_features": batch_global,
        "targets": batch_targets,
        "ids": ids_list,
    }
