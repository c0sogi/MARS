import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import compute_pbc_distances, compute_inertia_eigenvalues

# -------------------------------------------------------------------------
# Helper Functions for Geometry Parsing & Feature Extraction
# -------------------------------------------------------------------------


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file.
    Returns:
        lattice_vectors (3x3 np.array)
        atom_types (list of str)
        coords (Nx3 np.array)
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    lattice_vectors = []
    atom_types = []
    coords = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            # format: atom x y z symbol
            coords.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    coords_arr = np.array(coords)
    if coords_arr.size == 0:
        coords_arr = coords_arr.reshape(0, 3)

    return np.array(lattice_vectors), atom_types, coords_arr


def get_lattice_params(lattice_vectors):
    """
    Computes lattice lengths (a, b, c) and angles (alpha, beta, gamma).
    """
    v1, v2, v3 = lattice_vectors
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)

    # Clip values to [-1, 1] to avoid numerical errors in arccos
    cos_alpha = np.clip(np.dot(v2, v3) / (b * c), -1.0, 1.0)
    cos_beta = np.clip(np.dot(v1, v3) / (a * c), -1.0, 1.0)
    cos_gamma = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)

    alpha = np.degrees(np.arccos(cos_alpha))
    beta = np.degrees(np.arccos(cos_beta))
    gamma = np.degrees(np.arccos(cos_gamma))

    return np.array([a, b, c, alpha, beta, gamma])


def get_cell_volume(lattice_vectors):
    return np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )


def process_geometry(file_path, atom_type_map, k_neighbors):
    """
    Extracts atomic and global features for a single material.
    """
    lattice, types, coords = parse_xyz(file_path)
    num_atoms = len(types)

    # 1. Centering
    if num_atoms > 0:
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
    else:
        centered_coords = coords

    # 2. PBC Distances and Vectors
    # distances: (N, N), diff_vectors: (N, N, 3)
    distances, diff_vectors = compute_pbc_distances(centered_coords, lattice)

    # Mask self-distance (diagonal is 0) with infinity for sorting
    np.fill_diagonal(distances, np.inf)

    # 3. Atomic Features Construction
    atomic_features = []

    for i in range(num_atoms):
        # A. Atom Identity One-Hot
        atom_type_idx = atom_type_map.get(types[i])
        atom_oh = np.zeros(len(atom_type_map))
        atom_oh[atom_type_idx] = 1.0

        # B. Nearest Neighbor (NN) Info
        # Find index of min distance
        nn_idx = np.argmin(distances[i])
        nn_dist = distances[i, nn_idx]
        nn_type = types[nn_idx]
        nn_type_idx = atom_type_map.get(nn_type)
        nn_oh = np.zeros(len(atom_type_map))
        nn_oh[nn_type_idx] = 1.0

        # C. Local Inertia Eigenvalues
        # Get indices of K nearest neighbors
        # If N-1 < K, take all N-1 neighbors
        k_eff = min(k_neighbors, num_atoms - 1)
        if k_eff > 0:
            # argsort returns indices of smallest elements
            nearest_indices = np.argsort(distances[i])[:k_eff]
            # Get relative vectors to these neighbors
            neighbor_vecs = diff_vectors[i, nearest_indices, :]  # (K, 3)
            eigvals = compute_inertia_eigenvalues(neighbor_vecs)
        else:
            eigvals = np.zeros(3)

        # Concatenate: Atom_OH (4) + NN_OH (4) + Coords (3) + NN_Dist (1) + Eigs (3) = 15
        feat_vec = np.concatenate(
            [atom_oh, nn_oh, centered_coords[i], [nn_dist], eigvals]
        )
        atomic_features.append(feat_vec)

    if len(atomic_features) == 0:
        atomic_features = np.zeros((0, Config.ATOMIC_FEATURE_DIM), dtype=np.float32)
    else:
        atomic_features = np.array(atomic_features, dtype=np.float32)

    # 4. Global Features Construction
    # Lattice params (6)
    lat_params = get_lattice_params(lattice)
    # Volume (1)
    vol = get_cell_volume(lattice)
    # Density (1)
    density = num_atoms / vol
    # Stoichiometry (3 for Al, Ga, In)
    counts = {t: 0 for t in ["Al", "Ga", "In"]}
    for t in types:
        if t in counts:
            counts[t] += 1

    # Fraction of total atoms
    stoich = [counts[t] / num_atoms for t in ["Al", "Ga", "In"]]

    # Total Atoms (1)
    total_atoms = num_atoms

    # Concatenate: Lat (6) + Vol (1) + Dens (1) + Stoich (3) + Total (1) = 12
    global_features = np.concatenate(
        [lat_params, [vol], [density], stoich, [total_atoms]]
    ).astype(np.float32)

    return atomic_features, global_features


# -------------------------------------------------------------------------
# Dataset Processing & Caching
# -------------------------------------------------------------------------


def preprocess_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, processes all geometry files, and caches the result.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "atomic_features": data["atomic_features"],
                "global_features": data["global_features"],
                "targets": data["targets"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing dataset from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    ids = []
    atomic_features_list = []
    global_features_list = []
    targets_list = []

    atom_type_map = {t: i for i, t in enumerate(Config.ATOM_TYPES)}

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Process geometry
        af, gf = process_geometry(file_path, atom_type_map, Config.K_NEIGHBORS)

        ids.append(row["id"])
        atomic_features_list.append(af)
        global_features_list.append(gf)

        # Process targets if available
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            # Log transform targets: log(1 + y)
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets_list.append(np.array([t1, t2], dtype=np.float32))
        else:
            # Dummy targets for test set
            targets_list.append(np.array([0.0, 0.0], dtype=np.float32))

    # Convert to object arrays for variable length atomic features
    atomic_features_arr = np.array(atomic_features_list, dtype=object)
    global_features_arr = np.array(global_features_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    ids_arr = np.array(ids, dtype=np.int32)

    # Save to cache
    np.savez_compressed(
        cache_path,
        ids=ids_arr,
        atomic_features=atomic_features_arr,
        global_features=global_features_arr,
        targets=targets_arr,
    )
    print(f"Data cached to {cache_path}")

    return {
        "ids": ids_arr,
        "atomic_features": atomic_features_arr,
        "global_features": global_features_arr,
        "targets": targets_arr,
    }


# -------------------------------------------------------------------------
# Scaling
# -------------------------------------------------------------------------


def fit_and_save_scalers(train_data, scaler_path):
    """
    Computes mean and std for continuous features from training data.
    Atomic Continuous Indices: 8, 9, 10 (Coords), 11 (Dist), 12, 13, 14 (Eigs)
    Global Continuous Indices: All 0-11.
    """
    # Flatten atomic features to compute stats
    all_atomic = np.concatenate(train_data["atomic_features"], axis=0)

    # Select continuous columns
    # 0-3: Atom OH, 4-7: NN OH. 8-14: Continuous
    atomic_cont = all_atomic[:, 8:]

    atomic_mean = np.mean(atomic_cont, axis=0)
    atomic_std = np.std(atomic_cont, axis=0)
    # Avoid division by zero
    atomic_std[atomic_std == 0] = 1.0

    # Global features
    global_feats = train_data["global_features"]
    global_mean = np.mean(global_feats, axis=0)
    global_std = np.std(global_feats, axis=0)
    global_std[global_std == 0] = 1.0

    np.savez(
        scaler_path,
        atomic_mean=atomic_mean,
        atomic_std=atomic_std,
        global_mean=global_mean,
        global_std=global_std,
    )
    print(f"Scalers saved to {scaler_path}")
    return atomic_mean, atomic_std, global_mean, global_std


def load_scalers(scaler_path):
    data = np.load(scaler_path)
    return (
        data["atomic_mean"],
        data["atomic_std"],
        data["global_mean"],
        data["global_std"],
    )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class MaterialDataset(Dataset):
    def __init__(self, data_dict, scalers=None):
        """
        Args:
            data_dict: Dictionary containing ids, atomic_features, global_features, targets.
            scalers: Tuple (atomic_mean, atomic_std, global_mean, global_std) or None.
        """
        self.ids = data_dict["ids"]
        self.atomic_features = data_dict["atomic_features"]
        self.global_features = data_dict["global_features"]
        self.targets = data_dict["targets"]
        self.scalers = scalers

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Copy to avoid modifying the cached array in memory
        af = self.atomic_features[idx].copy()
        gf = self.global_features[idx].copy()
        target = self.targets[idx].copy()
        mat_id = self.ids[idx]

        # Apply scaling if provided
        if self.scalers is not None:
            a_mean, a_std, g_mean, g_std = self.scalers

            # Scale continuous atomic features (indices 8 to end)
            af[:, 8:] = (af[:, 8:] - a_mean) / a_std

            # Scale global features
            gf = (gf - g_mean) / g_std

        return {
            "id": mat_id,
            "atomic_features": torch.tensor(af, dtype=torch.float32),
            "global_features": torch.tensor(gf, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


# -------------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------------


def collate_fn(batch):
    """
    Pads atomic features to the maximum number of atoms in the batch.
    """
    ids = [item["id"] for item in batch]
    global_features = torch.stack([item["global_features"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])

    # Handle variable size atomic features
    atomic_features_list = [item["atomic_features"] for item in batch]
    lengths = [af.shape[0] for af in atomic_features_list]
    max_len = max(lengths)
    feature_dim = atomic_features_list[0].shape[1]

    # Create padded tensor
    batch_size = len(batch)
    padded_atomic = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, af in enumerate(atomic_features_list):
        l = lengths[i]
        padded_atomic[i, :l, :] = af
        mask[i, :l] = 1.0

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "atomic_features": padded_atomic,
        "global_features": global_features,
        "mask": mask,
        "target": targets,
    }


# -------------------------------------------------------------------------
# Data Loaders
# -------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get train, val, and test dataloaders.
    """
    # 1. Process Train Data
    train_data = preprocess_dataset(
        Config.TRAIN_META_PATH, Config.TRAIN_DATA_CACHE, load_cached_data
    )

    # 2. Fit or Load Scalers
    if load_cached_data and os.path.exists(Config.SCALERS_CACHE):
        print("Loading cached scalers...")
        scalers = load_scalers(Config.SCALERS_CACHE)
    else:
        print("Fitting scalers on training data...")
        scalers = fit_and_save_scalers(train_data, Config.SCALERS_CACHE)

    # 3. Process Val and Test Data
    val_data = preprocess_dataset(
        Config.VAL_META_PATH, Config.VAL_DATA_CACHE, load_cached_data
    )

    test_data = preprocess_dataset(
        Config.TEST_META_PATH, Config.TEST_DATA_CACHE, load_cached_data
    )

    # 4. Create Datasets
    train_dataset = MaterialDataset(train_data, scalers)
    val_dataset = MaterialDataset(val_data, scalers)
    test_dataset = MaterialDataset(test_data, scalers)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
