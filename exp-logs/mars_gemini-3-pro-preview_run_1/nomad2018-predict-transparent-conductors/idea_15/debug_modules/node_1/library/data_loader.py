import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.physics import (
    get_lattice_matrix,
    calculate_cell_volume,
    compute_pbc_interactions,
    get_local_potential,
    get_nearest_neighbor,
)


class Scaler:
    def __init__(self):
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None
        self.fitted = False

    def fit(self, atomic_data, global_data):
        # atomic_data: (Total_Atoms, Feat_Dim)
        # global_data: (Num_Samples, Global_Dim)

        # Atomic: 0-3 are one-hot (Al, Ga, In, O). 4-8 are continuous (x, y, z, nn_dist, potential).
        self.atomic_mean = np.mean(atomic_data, axis=0)
        self.atomic_std = np.std(atomic_data, axis=0)

        # Avoid division by zero and do not scale one-hot encoding
        self.atomic_std[self.atomic_std < 1e-6] = 1.0
        self.atomic_mean[:4] = 0.0
        self.atomic_std[:4] = 1.0

        # Global: All are continuous
        self.global_mean = np.mean(global_data, axis=0)
        self.global_std = np.std(global_data, axis=0)
        self.global_std[self.global_std < 1e-6] = 1.0

        self.fitted = True

    def transform_atomic(self, atomic_data):
        return (atomic_data - self.atomic_mean) / self.atomic_std

    def transform_global(self, global_data):
        return (global_data - self.global_mean) / self.global_std

    def save(self, path):
        np.savez(
            path,
            atomic_mean=self.atomic_mean,
            atomic_std=self.atomic_std,
            global_mean=self.global_mean,
            global_std=self.global_std,
        )

    def load(self, path):
        data = np.load(path)
        self.atomic_mean = data["atomic_mean"]
        self.atomic_std = data["atomic_std"]
        self.global_mean = data["global_mean"]
        self.global_std = data["global_std"]
        self.fitted = True


class MaterialDataset(Dataset):
    def __init__(self, atomic_feats_list, global_feats, targets, ids):
        self.atomic_feats_list = atomic_feats_list
        self.global_feats = global_feats
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.atomic_feats_list[idx]),
            torch.FloatTensor(self.global_feats[idx]),
            torch.FloatTensor(self.targets[idx]),
            self.ids[idx],
        )


def process_geometry_file(file_path, lattice_matrix):
    # Read geometry.xyz
    # Format lines: atom x y z Type
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    coords = []
    types = []

    with open(full_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "atom":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atom_type = parts[4]
                coords.append([x, y, z])
                types.append(atom_type)

    coords = np.array(coords)

    # 1. Center coordinates relative to unit cell centroid
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 2. Physics calculations with PBC
    dist_matrix = compute_pbc_interactions(centered_coords, lattice_matrix)
    potential = get_local_potential(dist_matrix)
    nn_dist = get_nearest_neighbor(dist_matrix)

    # 3. Features Construction
    # One-hot encoding
    num_atoms = len(types)
    one_hot = np.zeros((num_atoms, Config.NUM_ATOM_TYPES))
    for i, t in enumerate(types):
        if t in Config.ATOM_TYPES:
            one_hot[i, Config.ATOM_TYPES.index(t)] = 1.0

    # Concatenate: One-hot (4) + Centered Coords (3) + NN Dist (1) + Potential (1) = 9
    atomic_features = np.column_stack([one_hot, centered_coords, nn_dist, potential])

    return atomic_features


def prepare_data(metadata_path, split_name, scaler=None, load_cached_data=True):
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split_name} data from cache...")
        try:
            data = np.load(cache_file)
            flat_atomic = data["flat_atomic"]
            atom_counts = data["atom_counts"]
            global_feats = data["global_feats"]
            targets = data["targets"]
            ids = data["ids"]

            # Reconstruct list of atomic features from flattened array
            atomic_feats_list = []
            cursor = 0
            for count in atom_counts:
                atomic_feats_list.append(flat_atomic[cursor : cursor + count])
                cursor += count

            return atomic_feats_list, global_feats, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {split_name} data...")
    df = pd.read_csv(metadata_path)

    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    for _, row in df.iterrows():
        # Global Features Extraction
        a, b, c = (
            row["lattice_vector_1_ang"],
            row["lattice_vector_2_ang"],
            row["lattice_vector_3_ang"],
        )
        alpha, beta, gamma = (
            row["lattice_angle_alpha_degree"],
            row["lattice_angle_beta_degree"],
            row["lattice_angle_gamma_degree"],
        )

        lattice_matrix = get_lattice_matrix(a, b, c, alpha, beta, gamma)
        vol = calculate_cell_volume(lattice_matrix)

        n_atoms = row["number_of_total_atoms"]
        density = n_atoms / vol

        stoich = [
            row["percent_atom_al"],
            row["percent_atom_ga"],
            row["percent_atom_in"],
        ]

        # Global vector: [a, b, c, alpha, beta, gamma, vol, density, al, ga, in, total] (12 dims)
        g_feat = [a, b, c, alpha, beta, gamma, vol, density] + stoich + [n_atoms]
        global_feats_list.append(g_feat)

        # Atomic Features Extraction
        a_feat = process_geometry_file(row["file_path"], lattice_matrix)
        atomic_feats_list.append(a_feat)

        # Targets
        if "formation_energy_ev_natom" in row:
            # Log transform targets: log(1 + y) to match RMSLE metric
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets_list.append([t1, t2])
        else:
            targets_list.append([0.0, 0.0])  # Dummy for test

        ids_list.append(row["id"])

    global_feats = np.array(global_feats_list)
    targets = np.array(targets_list)
    ids = np.array(ids_list)

    # Scaling
    if scaler:
        if not scaler.fitted:
            # Flatten atomic features for fitting statistics
            flat_atomic = np.vstack(atomic_feats_list)
            scaler.fit(flat_atomic, global_feats)
            scaler.save(Config.SCALER_PATH)

        # Apply transform
        global_feats = scaler.transform_global(global_feats)
        atomic_feats_list = [scaler.transform_atomic(x) for x in atomic_feats_list]

    # Save to cache (flatten atomic list for npz storage)
    flat_atomic = np.vstack(atomic_feats_list)
    atom_counts = np.array([len(x) for x in atomic_feats_list])

    np.savez(
        cache_file,
        flat_atomic=flat_atomic,
        atom_counts=atom_counts,
        global_feats=global_feats,
        targets=targets,
        ids=ids,
    )

    return atomic_feats_list, global_feats, targets, ids


def collate_batch(batch):
    # batch is list of tuples: (atomic, global, target, id)
    atomic_list, global_list, target_list, id_list = zip(*batch)

    batch_size = len(batch)
    # Find max number of atoms in this batch for padding
    max_atoms = max([x.shape[0] for x in atomic_list])
    feat_dim = atomic_list[0].shape[1]

    # Pad atomic features
    padded_atomic = torch.zeros(batch_size, max_atoms, feat_dim)
    mask = torch.zeros(batch_size, max_atoms)

    for i, atoms in enumerate(atomic_list):
        n = atoms.shape[0]
        padded_atomic[i, :n, :] = atoms
        mask[i, :n] = 1.0

    stacked_global = torch.stack(global_list)
    stacked_targets = torch.stack(target_list)
    stacked_ids = torch.tensor(id_list)

    return padded_atomic, stacked_global, mask, stacked_targets, stacked_ids


def get_loaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    scaler = Scaler()

    # Check if scaler exists to load it, ensuring consistent scaling across runs
    if load_cached_data and os.path.exists(Config.SCALER_PATH):
        scaler.load(Config.SCALER_PATH)

    # Train
    train_atomic, train_global, train_targets, train_ids = prepare_data(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        "train",
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    # Val
    val_atomic, val_global, val_targets, val_ids = prepare_data(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        "val",
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    # Test
    test_atomic, test_global, test_targets, test_ids = prepare_data(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        "test",
        scaler=scaler,
        load_cached_data=load_cached_data,
    )

    train_ds = MaterialDataset(train_atomic, train_global, train_targets, train_ids)
    val_ds = MaterialDataset(val_atomic, val_global, val_targets, val_ids)
    test_ds = MaterialDataset(test_atomic, test_global, test_targets, test_ids)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    return train_loader, val_loader, test_loader
