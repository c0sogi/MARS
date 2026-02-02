import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    ATOM_MAP,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import (
    compute_cell_volume,
    cartesian_to_fractional,
    get_pbc_distances,
)


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.
    """
    lattice_vectors = []
    atoms = []
    coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            coords.append([float(x) for x in parts[1:4]])
            atoms.append(parts[4])

    return np.array(lattice_vectors), np.array(atoms), np.array(coords)


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Processes the dataset: parses geometry files, computes features, and caches the result.
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            list(data["atomic_features"]),
            data["global_features"],
            data["targets"],
            data["ids"],
        )

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    all_global_features = []
    all_targets = []
    all_ids = []

    for idx, row in df.iterrows():
        # 1. Parse Geometry
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)
        lattice_vectors, atom_types, cart_coords = parse_xyz(full_path)

        # 2. Atomic Features
        # Identity (One-hot)
        n_atoms = len(atom_types)
        one_hot = np.zeros((n_atoms, 4))  # Al, Ga, In, O
        for i, atom in enumerate(atom_types):
            if atom in ATOM_MAP:
                one_hot[i, ATOM_MAP[atom]] = 1.0

        # Centered Cartesian
        centroid = np.mean(cart_coords, axis=0)
        centered_coords = cart_coords - centroid

        # Fractional Coordinates
        frac_coords = cartesian_to_fractional(cart_coords, lattice_vectors)

        # PBC Nearest Neighbor Distance
        # This can be computationally expensive, so it's good we cache it
        nn_dists = get_pbc_distances(cart_coords, lattice_vectors)

        # Combine Atomic Features
        # [One-hot (4), Centered (3), Fractional (3), NN Dist (1)] -> Dim 11
        atomic_feats = np.concatenate(
            [one_hot, centered_coords, frac_coords, nn_dists], axis=1
        )
        all_atomic_features.append(atomic_feats.astype(np.float32))

        # 3. Global Features
        # Lattice lengths and angles
        lat_params = np.array(
            [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
            ]
        )

        # Volume and Density
        vol = compute_cell_volume(
            row["lattice_vector_1_ang"],
            row["lattice_vector_2_ang"],
            row["lattice_vector_3_ang"],
            row["lattice_angle_alpha_degree"],
            row["lattice_angle_beta_degree"],
            row["lattice_angle_gamma_degree"],
        )
        density = row["number_of_total_atoms"] / vol if vol > 1e-6 else 0.0

        # Stoichiometry
        stoich = np.array(
            [
                row["percent_atom_al"],
                row["percent_atom_ga"],
                row["percent_atom_in"],
                1.0
                - (
                    row["percent_atom_al"]
                    + row["percent_atom_ga"]
                    + row["percent_atom_in"]
                ),  # Oxygen approx
            ]
        )

        # Combine Global Features
        # [Lattice (6), Volume (1), Density (1), Stoich (4)] -> Dim 12
        global_feats = np.concatenate([lat_params, [vol, density], stoich])
        all_global_features.append(global_feats.astype(np.float32))

        # 4. Targets
        if "formation_energy_ev_natom" in row:
            targets = np.array(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )
            # Log transform targets: log(1 + y)
            # Ensure non-negative before log (bandgap can be small but usually positive, formation energy can be 0)
            targets = np.log1p(np.maximum(targets, 0))
        else:
            targets = np.array([np.nan, np.nan])  # Placeholder for test

        all_targets.append(targets.astype(np.float32))
        all_ids.append(row["id"])

    # Convert to numpy arrays (object array for variable length atomic features)
    all_atomic_features = np.array(all_atomic_features, dtype=object)
    all_global_features = np.array(all_global_features, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    all_ids = np.array(all_ids, dtype=np.int32)

    # Save to cache
    np.savez_compressed(
        cache_path,
        atomic_features=all_atomic_features,
        global_features=all_global_features,
        targets=all_targets,
        ids=all_ids,
    )
    print(f"Saved processed data to {cache_path}")

    return list(all_atomic_features), all_global_features, all_targets, all_ids


class MaterialDataset(Dataset):
    def __init__(self, atomic_features, global_features, targets, ids, scaler=None):
        self.atomic_features = atomic_features
        self.global_features = global_features
        self.targets = targets
        self.ids = ids
        self.scaler = scaler

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get raw data
        atom_f = self.atomic_features[idx]  # (N, 11)
        glob_f = self.global_features[idx]  # (12,)
        target = self.targets[idx]  # (2,)
        id_val = self.ids[idx]

        # Apply scaling if provided
        if self.scaler:
            # Scale Atomic Features (skip one-hot first 4 columns)
            # Continuous atomic cols: 4:11
            atom_continuous = atom_f[:, 4:]
            atom_continuous = (
                atom_continuous - self.scaler["atomic_mean"]
            ) / self.scaler["atomic_std"]
            atom_f = np.concatenate([atom_f[:, :4], atom_continuous], axis=1)

            # Scale Global Features
            glob_f = (glob_f - self.scaler["global_mean"]) / self.scaler["global_std"]

        return (
            torch.tensor(atom_f, dtype=torch.float32),
            torch.tensor(glob_f, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(id_val, dtype=torch.long),
        )


def collate_materials(batch):
    """
    Custom collate function to handle variable number of atoms.
    """
    atomic_feats, global_feats, targets, ids = zip(*batch)

    # Pad atomic features
    # lengths = [f.shape[0] for f in atomic_feats]
    padded_atomic = pad_sequence(atomic_feats, batch_first=True, padding_value=0.0)

    # Create mask (True for real atoms, False for padding)
    # Shape: (Batch, Max_Atoms)
    mask = padded_atomic.sum(dim=-1) != 0
    # Note: Sum might be 0 if features are 0, but one-hot ensures sum >= 1 for valid atoms.
    # A safer mask derivation:
    batch_size = len(atomic_feats)
    max_len = padded_atomic.shape[1]
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, f in enumerate(atomic_feats):
        mask[i, : f.shape[0]] = True

    # Stack other tensors
    global_feats = torch.stack(global_feats)
    targets = torch.stack(targets)
    ids = torch.stack(ids)

    return {
        "atomic_features": padded_atomic,
        "mask": mask,
        "global_features": global_feats,
        "targets": targets,
        "ids": ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    Calculates normalization statistics from the training set.
    """
    # 1. Process/Load all splits
    train_data = process_dataset(TRAIN_METADATA, "train_data", load_cached_data)
    val_data = process_dataset(VAL_METADATA, "val_data", load_cached_data)
    test_data = process_dataset(TEST_METADATA, "test_data", load_cached_data)

    # 2. Compute Statistics on Training Data
    print("Computing normalization statistics from training data...")

    # Global features stats
    train_global = train_data[1]
    global_mean = np.mean(train_global, axis=0)
    global_std = np.std(train_global, axis=0)
    # Handle constant columns to avoid division by zero
    global_std[global_std < 1e-6] = 1.0

    # Atomic features stats (continuous only: indices 4 to 11)
    # Concatenate all atomic feature arrays to compute global mean/std
    train_atomic_concat = np.concatenate(train_data[0], axis=0)
    atomic_continuous = train_atomic_concat[:, 4:]
    atomic_mean = np.mean(atomic_continuous, axis=0)
    atomic_std = np.std(atomic_continuous, axis=0)
    atomic_std[atomic_std < 1e-6] = 1.0

    scaler = {
        "global_mean": global_mean,
        "global_std": global_std,
        "atomic_mean": atomic_mean,
        "atomic_std": atomic_std,
    }

    # 3. Create Datasets
    train_dataset = MaterialDataset(*train_data, scaler=scaler)
    val_dataset = MaterialDataset(*val_data, scaler=scaler)
    test_dataset = MaterialDataset(*test_data, scaler=scaler)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
