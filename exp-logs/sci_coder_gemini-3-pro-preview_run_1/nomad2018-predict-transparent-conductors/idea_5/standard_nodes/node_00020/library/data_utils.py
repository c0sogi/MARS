import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    ATOM_TYPES,
    K_NEIGHBORS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    INPUT_DIR,
    WORKING_DIR,
)


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols.
        coords (np.ndarray): Nx3 array of atomic coordinates.
    """
    lattice_vectors = []
    atom_types = []
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
            atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(coords)


def compute_topology_fingerprint(coords, k_neighbors):
    """
    Computes the local topology fingerprint for each atom based on sorted distances
    to the K nearest neighbors.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        k_neighbors (int): Number of neighbors to include.

    Returns:
        fingerprints (np.ndarray): NxK array of sorted distances.
    """
    n_atoms = coords.shape[0]
    # Compute pairwise distance matrix (NxN)
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> norm -> (N, N)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))

    fingerprints = []
    for i in range(n_atoms):
        # Get distances for atom i, sort them
        # Exclude the first one (distance to self, which is 0.0)
        d_sorted = np.sort(dists[i])[1:]

        # If we have enough neighbors, take the first K
        if len(d_sorted) >= k_neighbors:
            fp = d_sorted[:k_neighbors]
        else:
            # Pad with a large value if not enough neighbors (unlikely in crystals but safe)
            padding = np.full(k_neighbors - len(d_sorted), 100.0)
            fp = np.concatenate([d_sorted, padding])

        fingerprints.append(fp)

    return np.array(fingerprints, dtype=np.float32)


def get_lattice_params(lattice_vectors):
    """
    Calculates lattice lengths and angles from lattice vectors.
    """
    a_vec = lattice_vectors[0]
    b_vec = lattice_vectors[1]
    c_vec = lattice_vectors[2]

    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    # Angles in degrees
    # alpha: angle between b and c
    alpha_rad = np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0))
    # beta: angle between a and c
    beta_rad = np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0))
    # gamma: angle between a and b
    gamma_rad = np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0))

    alpha = np.degrees(alpha_rad)
    beta = np.degrees(beta_rad)
    gamma = np.degrees(gamma_rad)

    return [a, b, c], [alpha, beta, gamma]


def process_data(metadata_path, cache_path, load_cached_data=True, max_samples=None):
    """
    Processes the dataset: parses geometry, computes features, and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the .npz cache.
        load_cached_data (bool): Whether to try loading from cache.
        max_samples (int): Limit number of samples for debugging.

    Returns:
        dict: Contains arrays for global_features, atomic_features, atom_counts, targets, ids.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path)
            return {
                "global_features": data["global_features"],
                "atomic_features_flat": data["atomic_features_flat"],
                "atom_counts": data["atom_counts"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    if max_samples is not None:
        df = df.iloc[:max_samples]

    global_features_list = []
    atomic_features_flat_list = []
    atom_counts_list = []
    targets_list = []
    ids_list = []

    # Mapping for one-hot encoding
    type_map = {atype: i for i, atype in enumerate(ATOM_TYPES)}

    for idx, row in df.iterrows():
        # Path relative to input dir
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}, skipping.")
            continue

        # Parse geometry
        lat_vecs, atom_types_list, coords = parse_xyz(full_path)
        n_atoms = len(atom_types_list)

        # --- Global Features ---
        # 1. Lattice lengths and angles
        lengths, angles = get_lattice_params(lat_vecs)

        # 2. Volume (scalar triple product)
        volume = np.abs(np.dot(lat_vecs[0], np.cross(lat_vecs[1], lat_vecs[2])))

        # 3. Density
        density = n_atoms / volume

        # 4. Composition (Al, Ga, In)
        # Note: The metadata CSV already has percent_atom_X, but we compute to be self-contained or verify
        # We will use the counts from the parsed file to be safe
        counts = {t: 0 for t in ATOM_TYPES}
        for t in atom_types_list:
            if t in counts:
                counts[t] += 1
        compositions = [counts[t] / n_atoms for t in ["Al", "Ga", "In"]]

        # Combine global features: [len_a, len_b, len_c, ang_a, ang_b, ang_c, comp_Al, comp_Ga, comp_In, vol, dens]
        g_feat = lengths + angles + compositions + [volume, density]
        global_features_list.append(g_feat)

        # --- Atomic Features ---
        # 1. One-hot encoding
        one_hot = np.zeros((n_atoms, len(ATOM_TYPES)), dtype=np.float32)
        for i, t in enumerate(atom_types_list):
            if t in type_map:
                one_hot[i, type_map[t]] = 1.0

        # 2. Centered Coordinates
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # 3. Topological Fingerprint
        fingerprints = compute_topology_fingerprint(coords, K_NEIGHBORS)

        # Concatenate: [OneHot(4) | Coords(3) | Fingerprint(K)]
        a_feat = np.concatenate([one_hot, centered_coords, fingerprints], axis=1)
        atomic_features_flat_list.append(a_feat)
        atom_counts_list.append(n_atoms)

        # --- Targets ---
        # Check if targets exist (train/val) or placeholder (test)
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            # Apply log(1+y) transformation
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets_list.append([t1, t2])
        else:
            targets_list.append([0.0, 0.0])  # Dummy for test

        ids_list.append(row["id"])

    # Convert to numpy arrays
    global_features = np.array(global_features_list, dtype=np.float32)
    # Flatten atomic features for storage
    atomic_features_flat = np.concatenate(atomic_features_flat_list, axis=0).astype(
        np.float32
    )
    atom_counts = np.array(atom_counts_list, dtype=np.int32)
    targets = np.array(targets_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=np.int32)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        global_features=global_features,
        atomic_features_flat=atomic_features_flat,
        atom_counts=atom_counts,
        targets=targets,
        ids=ids,
    )
    print(f"Data saved to {cache_path}")

    return {
        "global_features": global_features,
        "atomic_features_flat": atomic_features_flat,
        "atom_counts": atom_counts,
        "targets": targets,
        "ids": ids,
    }


class CrystalDataset(Dataset):
    def __init__(self, data_dict, scaler=None):
        """
        Args:
            data_dict (dict): Output from process_data.
            scaler (object, optional): Object with transform method for normalization.
        """
        self.global_features = torch.tensor(
            data_dict["global_features"], dtype=torch.float32
        )
        self.atom_counts = torch.tensor(data_dict["atom_counts"], dtype=torch.long)
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        self.ids = torch.tensor(data_dict["ids"], dtype=torch.long)

        # Reconstruct list of atomic feature tensors
        flat_atomic = torch.tensor(
            data_dict["atomic_features_flat"], dtype=torch.float32
        )
        self.atomic_features = []
        start = 0
        for count in self.atom_counts:
            end = start + count.item()
            self.atomic_features.append(flat_atomic[start:end])
            start = end

        # Apply scaling if provided
        if scaler is not None:
            self.global_features = scaler.transform_global(self.global_features)
            # Scale atomic features (excluding one-hot part usually, but simple z-score on all is robust enough for MLPs)
            # Note: One-hot is indices 0-3. Coords 4-6. Fingerprint 7+.
            # We usually don't scale one-hot.
            for i in range(len(self.atomic_features)):
                self.atomic_features[i] = scaler.transform_atomic(
                    self.atomic_features[i]
                )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "global_features": self.global_features[idx],
            "atomic_features": self.atomic_features[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def collate_fn(batch):
    """
    Collates a batch of variable-size crystal graphs.
    """
    global_feats = torch.stack([b["global_features"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    ids = torch.stack([b["id"] for b in batch])

    # Handle variable atomic features
    atomic_feats_list = [b["atomic_features"] for b in batch]
    lengths = [f.shape[0] for f in atomic_feats_list]
    max_len = max(lengths)
    feature_dim = atomic_feats_list[0].shape[1]

    # Pad
    padded_atomic = torch.zeros(len(batch), max_len, feature_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (feat, length) in enumerate(zip(atomic_feats_list, lengths)):
        padded_atomic[i, :length, :] = feat
        mask[i, :length] = True

    return {
        "global_features": global_feats,
        "atomic_features": padded_atomic,
        "mask": mask,
        "targets": targets,
        "ids": ids,
    }


class StandardScaler:
    """
    Simple standard scaler for PyTorch tensors.
    """

    def __init__(self):
        self.g_mean = None
        self.g_std = None
        self.a_mean = None
        self.a_std = None

    def fit(self, global_feats, atomic_feats_flat):
        # Global
        self.g_mean = global_feats.mean(dim=0)
        self.g_std = global_feats.std(dim=0)
        # Avoid div by zero
        self.g_std[self.g_std < 1e-6] = 1.0

        # Atomic: Only scale continuous parts (Coords + Fingerprint)
        # Indices 0-3 are One-Hot. 4-6 Coords. 7+ Fingerprint.
        # We will scale indices 4 onwards.
        continuous_atomic = atomic_feats_flat[:, 4:]
        self.a_mean = continuous_atomic.mean(dim=0)
        self.a_std = continuous_atomic.std(dim=0)
        self.a_std[self.a_std < 1e-6] = 1.0

    def transform_global(self, g_feats):
        return (g_feats - self.g_mean) / self.g_std

    def transform_atomic(self, a_feats):
        # a_feats is (N_atoms, Dim)
        # Don't touch one-hot (0:4)
        one_hot = a_feats[:, :4]
        continuous = a_feats[:, 4:]
        scaled_continuous = (continuous - self.a_mean) / self.a_std
        return torch.cat([one_hot, scaled_continuous], dim=1)
