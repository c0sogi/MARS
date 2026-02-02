import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import compute_pbc_distance_matrix


class Scaler:
    """
    Simple Standard Scaler for numpy arrays.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0)
        # Avoid division by zero
        self.scale[self.scale < 1e-8] = 1.0

    def transform(self, data):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted.")
        return (data - self.mean) / self.scale

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.scale = data["scale"]


def parse_xyz(file_path):
    """
    Parses the specific XYZ format provided in the dataset.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple: (lattice_vectors (3x3 np.array), atoms (list of (species, position)))
    """
    lattice_vectors = []
    atoms = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                pos = [float(x) for x in parts[1:4]]
                species = parts[4]
                atoms.append((species, np.array(pos)))

    return np.array(lattice_vectors), atoms


def get_atomic_one_hot(species):
    """
    Returns a one-hot encoding for the atomic species.
    """
    mapping = {s: i for i, s in enumerate(Config.ATOMIC_SPECIES)}
    one_hot = np.zeros(Config.NUM_SPECIES)
    if species in mapping:
        one_hot[mapping[species]] = 1.0
    return one_hot


def compute_chem_resolved_distances(positions, species_list, lattice_vectors):
    """
    Computes the minimum distance from each atom to the nearest neighbor
    of each specific chemical species, respecting PBC.

    Args:
        positions (np.ndarray): (N, 3) atomic positions.
        species_list (list): List of species strings corresponding to positions.
        lattice_vectors (np.ndarray): (3, 3) lattice vectors.

    Returns:
        np.ndarray: (N, 4) matrix of chemically resolved distances.
    """
    N = len(positions)
    # Calculate full distance matrix with PBC
    dist_matrix = compute_pbc_distance_matrix(positions, lattice_vectors)

    # Mask self-distances (diagonal) with infinity so they aren't selected as min
    np.fill_diagonal(dist_matrix, np.inf)

    # Initialize with the "infinite" placeholder value
    chem_dists = np.full((N, Config.NUM_SPECIES), Config.INFINITE_DISTANCE_VAL)

    # Map species to their indices in the molecule
    species_indices = {s: [] for s in Config.ATOMIC_SPECIES}
    for idx, s in enumerate(species_list):
        if s in species_indices:
            species_indices[s].append(idx)

    # For each atom, find min distance to each species type
    for i in range(N):
        for s_idx, s_name in enumerate(Config.ATOMIC_SPECIES):
            target_indices = species_indices[s_name]
            if len(target_indices) > 0:
                # Get distances from atom i to all atoms of species s_name
                dists_to_species = dist_matrix[i, target_indices]
                min_dist = np.min(dists_to_species)

                # Update if a valid neighbor exists
                if not np.isinf(min_dist):
                    chem_dists[i, s_idx] = min_dist

    return chem_dists


def compute_global_features(lattice_vectors, atoms):
    """
    Computes macroscopic features for the Global Stream.

    Returns:
        np.ndarray: 12-dimensional feature vector.
    """
    # 1. Lattice lengths (a, b, c)
    lengths = np.linalg.norm(lattice_vectors, axis=1)

    # 2. Lattice angles (alpha, beta, gamma)
    # alpha: angle between b and c
    # beta: angle between a and c
    # gamma: angle between a and b
    def get_angle(v1, v2):
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        # Clip to avoid numerical errors outside [-1, 1]
        cos_angle = np.clip(dot / norms, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    alpha = get_angle(lattice_vectors[1], lattice_vectors[2])
    beta = get_angle(lattice_vectors[0], lattice_vectors[2])
    gamma = get_angle(lattice_vectors[0], lattice_vectors[1])

    # 3. Unit Cell Volume
    # V = |a . (b x c)|
    volume = np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )

    # 4. Total Atoms
    n_atoms = len(atoms)

    # 5. Atomic Density
    density = n_atoms / volume

    # 6. Stoichiometry (fractions of Al, Ga, In)
    counts = {s: 0 for s in ["Al", "Ga", "In"]}
    for s, _ in atoms:
        if s in counts:
            counts[s] += 1

    # Avoid division by zero if n_atoms is 0 (should not happen)
    fracs = [counts[s] / n_atoms for s in ["Al", "Ga", "In"]]

    # Combine all features
    # [3 lengths, 3 angles, 1 volume, 1 density, 1 count, 3 fractions] = 12 dims
    return np.concatenate(
        [lengths, [alpha, beta, gamma], [volume, density, n_atoms], fracs]
    )


def process_dataset(metadata_path, load_cached_data=True, mode="train"):
    """
    Parses raw data, computes features, and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to try loading from cache.
        mode (str): 'train', 'val', or 'test' to determine cache filename.

    Returns:
        dict: Dictionary containing numpy arrays of inputs, indices, targets, and ids.
    """
    # Determine cache file path
    cache_filename = f"{mode}_data.npz"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        try:
            data = np.load(cache_path)
            return {
                "atomic_inputs": data["atomic_inputs"],
                "global_inputs": data["global_inputs"],
                "batch_indices": data["batch_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {mode} data from scratch...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    all_atomic_inputs = []
    all_global_inputs = []
    all_batch_indices = []
    all_targets = []
    all_ids = []

    # Processing loop
    # Use a counter for batch indices to ensure they are contiguous 0..N-1
    sample_idx = 0

    for _, row in df.iterrows():
        sample_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}, skipping.")
            continue

        # Parse geometry
        lattice_vectors, atoms = parse_xyz(full_path)
        positions = np.array([a[1] for a in atoms])
        species = [a[0] for a in atoms]
        n_atoms = len(atoms)

        # --- Atomic Stream Features ---
        # 1. One-hot encoding (N, 4)
        one_hots = np.array([get_atomic_one_hot(s) for s in species])

        # 2. Centered Coordinates (N, 3)
        centroid = np.mean(positions, axis=0)
        centered_coords = positions - centroid

        # 3. Chemically Resolved Distances (N, 4)
        chem_dists = compute_chem_resolved_distances(
            positions, species, lattice_vectors
        )

        # Combine Atomic Features (N, 11)
        atomic_feats = np.hstack([one_hots, centered_coords, chem_dists])

        # --- Global Stream Features ---
        # (12,)
        global_feats = compute_global_features(lattice_vectors, atoms)

        # --- Targets ---
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            target = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
        else:
            # Placeholder for test set
            target = [0.0, 0.0]

        # Append to lists
        all_atomic_inputs.append(atomic_feats)
        all_global_inputs.append(global_feats)
        all_batch_indices.append(np.full(n_atoms, sample_idx))
        all_targets.append(target)
        all_ids.append(sample_id)

        sample_idx += 1

    # Concatenate all data into numpy arrays
    if not all_atomic_inputs:
        raise ValueError("No data processed!")

    atomic_inputs_np = np.vstack(all_atomic_inputs).astype(np.float32)
    global_inputs_np = np.vstack(all_global_inputs).astype(np.float32)
    batch_indices_np = np.concatenate(all_batch_indices).astype(np.int64)
    targets_np = np.array(all_targets).astype(np.float32)
    ids_np = np.array(all_ids).astype(np.int64)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez(
        cache_path,
        atomic_inputs=atomic_inputs_np,
        global_inputs=global_inputs_np,
        batch_indices=batch_indices_np,
        targets=targets_np,
        ids=ids_np,
    )

    print(f"Processed {sample_idx} samples. Saved to {cache_path}")

    return {
        "atomic_inputs": atomic_inputs_np,
        "global_inputs": global_inputs_np,
        "batch_indices": batch_indices_np,
        "targets": targets_np,
        "ids": ids_np,
    }


def get_scalers(train_data):
    """
    Fits scalers on training data or loads them if they exist.
    """
    scaler_path = os.path.join(Config.WORKING_DIR, Config.CACHE_SCALERS)

    atomic_scaler = Scaler()
    global_scaler = Scaler()

    # We always fit on the provided train_data to ensure consistency with the current run
    # But we can save them for inference later
    print("Fitting scalers on training data...")

    # Fit atomic scaler (exclude one-hot encoding which are first 4 cols)
    # We scale coords (3) and distances (4) -> indices 4 to 11
    atomic_feats = train_data["atomic_inputs"]
    atomic_continuous = atomic_feats[:, 4:]
    atomic_scaler.fit(atomic_continuous)

    # Fit global scaler (all 12 dims are continuous/ordinal)
    global_feats = train_data["global_inputs"]
    global_scaler.fit(global_feats)

    # Save for future use (e.g. inference)
    np.savez(
        scaler_path,
        atomic_mean=atomic_scaler.mean,
        atomic_scale=atomic_scaler.scale,
        global_mean=global_scaler.mean,
        global_scale=global_scaler.scale,
    )

    return atomic_scaler, global_scaler


def load_scalers():
    """
    Loads pre-computed scalers from disk.
    """
    scaler_path = os.path.join(Config.WORKING_DIR, Config.CACHE_SCALERS)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler cache not found at {scaler_path}. Run training first."
        )

    data = np.load(scaler_path)

    atomic_scaler = Scaler()
    atomic_scaler.mean = data["atomic_mean"]
    atomic_scaler.scale = data["atomic_scale"]

    global_scaler = Scaler()
    global_scaler.mean = data["global_mean"]
    global_scaler.scale = data["global_scale"]

    return atomic_scaler, global_scaler


def scale_features(data_dict, atomic_scaler, global_scaler):
    """
    Applies scaling to the features in the data dictionary.
    Returns a NEW dictionary with scaled data.
    """
    atomic_inputs = data_dict["atomic_inputs"].copy()
    global_inputs = data_dict["global_inputs"].copy()

    # Scale atomic continuous features (indices 4:)
    atomic_inputs[:, 4:] = atomic_scaler.transform(atomic_inputs[:, 4:])

    # Scale global features
    global_inputs = global_scaler.transform(global_inputs)

    return {
        "atomic_inputs": atomic_inputs,
        "global_inputs": global_inputs,
        "batch_indices": data_dict["batch_indices"],
        "targets": data_dict["targets"],
        "ids": data_dict["ids"],
    }
