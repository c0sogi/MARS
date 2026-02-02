import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ase import Atoms
from library.config import Config
from library.utils import log_transform

# Maximum number of atoms to pad to (based on analysis max is 80)
MAX_ATOMS = 100


def parse_custom_xyz(file_path):
    """
    Parses the custom XYZ format provided in the dataset.
    Format:
    lattice_vector x y z
    lattice_vector x y z
    lattice_vector x y z
    atom x y z Symbol
    ...
    """
    lattice_vectors = []
    positions = []
    symbols = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                positions.append([float(x) for x in parts[1:4]])
                symbols.append(parts[4])

    cell = np.array(lattice_vectors)
    positions = np.array(positions)

    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def process_dataset(df, input_dir):
    """
    Processes the dataframe to extract features from XYZ files.
    Returns a dictionary of numpy arrays.
    """
    num_samples = len(df)

    # Pre-allocate arrays
    # atom_types: (N, MAX_ATOMS)
    # lattice_features: (N, 6) -> [a, b, c, alpha, beta, gamma]
    # mask: (N, MAX_ATOMS) -> 1 for atom, 0 for padding
    # targets: (N, 2) -> [formation_energy, bandgap]
    # ids: (N,)

    all_atom_types = np.zeros((num_samples, MAX_ATOMS), dtype=np.int64)
    all_lattice_features = np.zeros((num_samples, 6), dtype=np.float32)
    all_masks = np.zeros((num_samples, MAX_ATOMS), dtype=np.float32)
    all_targets = np.zeros((num_samples, 2), dtype=np.float32)
    all_ids = np.zeros((num_samples,), dtype=np.int64)

    atom_map = Config.ATOM_MAP

    for idx, (_, row) in enumerate(df.iterrows()):
        file_path = os.path.join(input_dir, row["file_path"])
        atoms = parse_custom_xyz(file_path)

        # 1. Atom Types
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(symbols)
        if n_atoms > MAX_ATOMS:
            # Truncate if larger (should not happen based on analysis)
            symbols = symbols[:MAX_ATOMS]
            n_atoms = MAX_ATOMS

        type_indices = [atom_map[s] for s in symbols]
        all_atom_types[idx, :n_atoms] = type_indices
        all_masks[idx, :n_atoms] = 1.0

        # 2. Lattice Parameters
        # cellpar returns [a, b, c, alpha, beta, gamma]
        # lengths in Angstrom, angles in degrees
        cell_params = atoms.cell.cellpar()
        all_lattice_features[idx, :] = cell_params

        # 3. Targets & ID
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            all_targets[idx, 0] = row["formation_energy_ev_natom"]
            all_targets[idx, 1] = row["bandgap_energy_ev"]

        all_ids[idx] = row["id"]

    return {
        "atom_types": all_atom_types,
        "lattice_features": all_lattice_features,
        "masks": all_masks,
        "targets": all_targets,
        "ids": all_ids,
    }


class CrystalDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, limit=None):
        super().__init__()
        self.mode = mode

        # Determine paths
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
            self.cache_path = Config.CACHE_TRAIN_DATA
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
            self.cache_path = Config.CACHE_VAL_DATA
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
            self.cache_path = Config.CACHE_TEST_DATA
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load or Process Data
        data_loaded = False
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached {mode} data from {self.cache_path}...")
                loaded = np.load(self.cache_path)
                self.atom_types = torch.from_numpy(loaded["atom_types"])
                self.lattice_features = torch.from_numpy(loaded["lattice_features"])
                self.masks = torch.from_numpy(loaded["masks"])
                self.targets = torch.from_numpy(loaded["targets"])
                self.ids = torch.from_numpy(loaded["ids"])
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}")

        if not data_loaded:
            print(f"Processing {mode} data from scratch...")
            df = pd.read_csv(self.csv_path)

            # Debug limit
            if limit is not None:
                df = df.head(limit)
            elif Config.DEBUG_DATA_LIMIT is not None:
                df = df.head(Config.DEBUG_DATA_LIMIT)

            processed = process_dataset(df, Config.INPUT_DIR)

            # Save to cache
            np.savez(self.cache_path, **processed)

            self.atom_types = torch.from_numpy(processed["atom_types"])
            self.lattice_features = torch.from_numpy(processed["lattice_features"])
            self.masks = torch.from_numpy(processed["masks"])
            self.targets = torch.from_numpy(processed["targets"])
            self.ids = torch.from_numpy(processed["ids"])

        # Apply Log Transformation to Targets if training/val
        if mode in ["train", "val"]:
            self.targets = log_transform(self.targets)

        # Simple scaling for lattice features (heuristic)
        # Lengths (0-25A) -> / 10.0
        # Angles (0-180) -> / 100.0
        self.lattice_features[:, :3] /= 10.0
        self.lattice_features[:, 3:] /= 100.0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "atom_types": self.atom_types[idx],  # (MAX_ATOMS,)
            "lattice_features": self.lattice_features[idx],  # (6,)
            "mask": self.masks[idx],  # (MAX_ATOMS,)
            "target": self.targets[idx],  # (2,)
            "id": self.ids[idx],  # (1,)
        }


def collate_batch(batch):
    """
    Collate function for DataLoader.
    Since we padded to MAX_ATOMS in the dataset, we can just stack.
    """
    atom_types = torch.stack([item["atom_types"] for item in batch])
    lattice_features = torch.stack([item["lattice_features"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    ids = torch.stack([item["id"] for item in batch])

    return {
        "atom_types": atom_types,
        "lattice_features": lattice_features,
        "mask": mask,
        "target": target,
        "id": ids,
    }
