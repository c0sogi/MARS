import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import (
    calculate_lattice_params,
    calculate_cell_volume,
    get_pbc_distances,
)


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material properties.
    """

    def __init__(
        self,
        atomic_features_flat,
        atomic_counts,
        global_features,
        targets,
        ids,
        atomic_scaler=None,
        global_scaler=None,
        is_train=False,
    ):
        self.atomic_features_flat = atomic_features_flat
        self.atomic_counts = atomic_counts
        self.global_features = global_features
        self.targets = targets
        self.ids = ids
        self.atomic_scaler = atomic_scaler
        self.global_scaler = global_scaler
        self.is_train = is_train

        # Pre-calculate cumulative indices for fast slicing
        self.cumulative_counts = np.concatenate(([0], np.cumsum(atomic_counts)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Extract atomic features for this crystal
        start_idx = self.cumulative_counts[idx]
        end_idx = self.cumulative_counts[idx + 1]

        atom_x = self.atomic_features_flat[start_idx:end_idx].copy()
        glob_x = self.global_features[idx].copy()

        # Apply scaling if scalers are provided
        if self.atomic_scaler:
            # Scale coords (cols 4-6) and NN dist (col 7)
            # One-hot encoding (cols 0-3) is typically not scaled or scaled differently.
            # Standard scaling one-hot is acceptable for neural nets to balance variance.
            if atom_x.shape[0] > 0:
                atom_x = self.atomic_scaler.transform(atom_x)
            else:
                atom_x = atom_x.reshape(0, Config.ATOMIC_FEATURE_DIM)

        if self.global_scaler:
            glob_x = self.global_scaler.transform(glob_x.reshape(1, -1)).flatten()

        # Targets (log transformed)
        y = np.zeros(Config.NUM_TARGETS, dtype=np.float32)
        if self.targets is not None:
            y = np.log1p(self.targets[idx])  # Log1p transformation

        return (
            torch.tensor(atom_x, dtype=torch.float32),
            torch.tensor(glob_x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            self.ids[idx],
        )


def collate_materials(batch):
    """
    Custom collate function to handle variable number of atoms.
    """
    batch_atomic_x = []
    batch_global_x = []
    batch_y = []
    batch_ids = []
    batch_indices = []

    for i, (atom_x, glob_x, y, id_) in enumerate(batch):
        batch_atomic_x.append(atom_x)
        batch_global_x.append(glob_x)
        batch_y.append(y)
        batch_ids.append(id_)
        # Create batch index for each atom (for scatter operations)
        batch_indices.append(torch.full((atom_x.shape[0],), i, dtype=torch.long))

    return (
        torch.cat(batch_atomic_x, dim=0),  # (Total_Atoms, Atomic_Dim)
        torch.stack(batch_global_x, dim=0),  # (Batch, Global_Dim)
        torch.stack(batch_y, dim=0),  # (Batch, Num_Targets)
        torch.cat(batch_indices, dim=0),  # (Total_Atoms,)
        batch_ids,
    )


def parse_geometry_file(file_path):
    """
    Parses a geometry.xyz file.
    Returns lattice matrix, atom types, and atom positions.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    with open(full_path, "r") as f:
        lines = f.readlines()

    lattice_vectors = []
    atom_types = []
    atom_positions = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            atom_positions.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    atom_positions_array = np.array(atom_positions)
    if atom_positions_array.ndim == 1:
        atom_positions_array = atom_positions_array.reshape(-1, 3)

    return np.array(lattice_vectors), atom_types, atom_positions_array


def process_dataset(metadata_path, cache_path, load_cached_data=True, debug_size=None):
    """
    Processes the dataset: parses geometry, calculates features, and caches result.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return (
                data["atomic_features_flat"],
                data["atomic_counts"],
                data["global_features"],
                data["targets"] if "targets" in data else None,
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if debug_size is not None:
        df = df.iloc[:debug_size]

    all_atomic_feats = []
    all_atomic_counts = []
    all_global_feats = []
    all_targets = []
    all_ids = []

    atom_map = Config.ATOM_MAP

    for idx, row in df.iterrows():
        # Parse geometry
        lattice_mat, atom_types, positions = parse_geometry_file(row["file_path"])

        # 1. Global Features
        # Lattice params
        lengths, angles = calculate_lattice_params(lattice_mat)
        # Volume
        vol = calculate_cell_volume(lattice_mat)
        # Density
        num_atoms = len(atom_types)
        density = num_atoms / vol if vol > 1e-6 else 0.0
        # Stoichiometry
        stoich = np.zeros(4)
        for at in atom_types:
            stoich[atom_map[at]] += 1
        if num_atoms > 0:
            stoich /= num_atoms

        global_vec = np.concatenate([lengths, angles, [vol, density], stoich])
        all_global_feats.append(global_vec)

        # 2. Atomic Features
        if num_atoms > 0:
            # Centering
            centroid = np.mean(positions, axis=0)
            centered_pos = positions - centroid

            # PBC Distances
            dist_matrix = get_pbc_distances(positions, lattice_mat)
            # Mask self-distance (0) with infinity to find nearest neighbor
            np.fill_diagonal(dist_matrix, np.inf)
            nn_dists = np.min(dist_matrix, axis=1).reshape(-1, 1)
        else:
            # Handle empty crystals (Cite debug_lesson_5)
            centered_pos = np.zeros((0, 3))
            nn_dists = np.zeros((0, 1))

        # One-hot encoding
        one_hot = np.zeros((num_atoms, 4))
        for i, at in enumerate(atom_types):
            one_hot[i, atom_map[at]] = 1.0

        # Combine: [One-hot(4), Centered_Pos(3), NN_Dist(1)]
        atomic_vecs = np.hstack([one_hot, centered_pos, nn_dists])

        all_atomic_feats.append(atomic_vecs)
        all_atomic_counts.append(num_atoms)

        # Targets
        if "formation_energy_ev_natom" in row:
            all_targets.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )

        all_ids.append(row["id"])

    # Flatten atomic features for storage
    flat_atomic_feats = np.vstack(all_atomic_feats)
    atomic_counts = np.array(all_atomic_counts)
    global_feats = np.array(all_global_feats)
    ids = np.array(all_ids)

    targets = None
    if all_targets:
        targets = np.array(all_targets)

    # Cache results
    save_dict = {
        "atomic_features_flat": flat_atomic_feats,
        "atomic_counts": atomic_counts,
        "global_features": global_feats,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez(cache_path, **save_dict)
    print(f"Data cached to {cache_path}")

    return flat_atomic_feats, atomic_counts, global_feats, targets, ids


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Main function to prepare DataLoaders.
    Handles processing, scaling, and loader creation.
    """
    # 1. Process Data
    train_data = process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data,
        debug_size,
    )
    val_data = process_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data, debug_size
    )
    test_data = process_dataset(
        Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, load_cached_data, debug_size
    )

    # Unpack Train
    train_atom_x, train_counts, train_glob_x, train_y, train_ids = train_data

    # 2. Fit Scalers on Training Data
    print("Fitting scalers on training data...")
    atomic_scaler = StandardScaler()
    if train_atom_x.shape[0] > 0:
        atomic_scaler.fit(train_atom_x)

    global_scaler = StandardScaler()
    global_scaler.fit(train_glob_x)

    # 3. Create Datasets
    train_dataset = MaterialDataset(
        *train_data,
        atomic_scaler=atomic_scaler,
        global_scaler=global_scaler,
        is_train=True,
    )

    val_dataset = MaterialDataset(
        *val_data,
        atomic_scaler=atomic_scaler,
        global_scaler=global_scaler,
        is_train=False,
    )

    test_dataset = MaterialDataset(
        *test_data,
        atomic_scaler=atomic_scaler,
        global_scaler=global_scaler,
        is_train=False,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
