import os
import numpy as np
import pandas as pd
from library.config import (
    ATOMIC_PROPERTIES,
    NEIGHBORS_K,
    WORKING_DIR,
    INPUT_DIR,
    TRAIN_CSV,
    TEST_CSV,
    METADATA_DIR,
    METADATA_TRAIN,
    METADATA_VAL,
    METADATA_TEST,
)


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors and atomic information.

    Args:
        file_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz')

    Returns:
        tuple: (lattice_vectors, atom_symbols, coords)
            - lattice_vectors: (3, 3) numpy array
            - atom_symbols: list of strings
            - coords: (N, 3) numpy array
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    lattice_vectors = []
    atoms = []
    coords = []

    with open(full_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            # Format: atom x y z symbol
            coords.append([float(x) for x in parts[1:4]])
            atoms.append(parts[4])

    # Cite debug_lesson_3: Enforce Consistent Rank When Creating NumPy Arrays from Potentially Empty Lists
    return np.array(lattice_vectors), atoms, np.array(coords).reshape(-1, 3)


def get_pbc_distances(coords, lattice):
    """
    Computes pairwise distances respecting periodic boundary conditions using
    the Minimum Image Convention via fractional coordinates.

    Args:
        coords (np.ndarray): (N, 3) Cartesian coordinates
        lattice (np.ndarray): (3, 3) Lattice vectors

    Returns:
        np.ndarray: (N, N) Distance matrix
    """
    # Cite debug_lesson_11: Explicitly Handle Zero-Cardinality Inputs
    if coords.shape[0] == 0:
        return np.zeros((0, 0))

    # Compute inverse lattice to convert to fractional coordinates
    try:
        inv_lattice = np.linalg.inv(lattice)
    except np.linalg.LinAlgError:
        # Fallback for singular lattice (should not happen in valid data)
        return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    # Convert to fractional coordinates: r_frac = r_cart @ inv_lattice
    frac_coords = coords @ inv_lattice

    # Compute differences in fractional space: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff_frac = frac_coords[:, None, :] - frac_coords[None, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    diff_frac = diff_frac - np.round(diff_frac)

    # Convert back to Cartesian: r_cart_diff = r_frac_diff @ lattice
    diff_cart = diff_frac @ lattice

    # Compute Euclidean distances
    dist_matrix = np.linalg.norm(diff_cart, axis=-1)

    return dist_matrix


def get_lce_features(atoms, dist_matrix):
    """
    Computes Local Chemical Environment features for each atom.
    Features: Inverse-distance weighted average of Mass, Radius, and Electronegativity
    of the K nearest neighbors.

    Args:
        atoms (list): List of atomic symbols
        dist_matrix (np.ndarray): (N, N) Pairwise distance matrix

    Returns:
        tuple: (lce_feats, nn_dists)
            - lce_feats: (N, 3) array [avg_mass, avg_radius, avg_neg]
            - nn_dists: (N, 1) array [distance to nearest neighbor]
    """
    n_atoms = len(atoms)

    # Cite debug_lesson_11: Explicitly Handle Zero-Cardinality Inputs
    if n_atoms == 0:
        return np.zeros((0, 3)), np.zeros((0, 1))

    lce_feats = np.zeros((n_atoms, 3))
    nn_dists = np.zeros((n_atoms, 1))

    # Pre-fetch properties for all atoms in the cell: [Mass, Radius, Electronegativity]
    # Default to [0, 0, 0] if element not found (should not happen for Al, Ga, In, O)
    props = np.array([ATOMIC_PROPERTIES.get(s, [0.0, 0.0, 0.0]) for s in atoms])

    for i in range(n_atoms):
        # Get distances from atom i to all others
        dists = dist_matrix[i]

        # Sort indices by distance
        # The first index is always i itself (dist=0), so we skip it
        sorted_indices = np.argsort(dists)

        # Determine K: use NEIGHBORS_K or N-1 if the cell is small
        k = min(NEIGHBORS_K, n_atoms - 1)
        if k <= 0:
            continue

        # Select K nearest neighbors
        neighbor_indices = sorted_indices[1 : k + 1]
        neighbor_dists = dists[neighbor_indices]

        # Feature: Distance to the nearest neighbor
        nn_dists[i] = neighbor_dists[0]

        # Calculate weights: 1 / distance
        # Add small epsilon to avoid division by zero (though neighbor dist > 0)
        weights = 1.0 / (neighbor_dists + 1e-6)
        total_weight = np.sum(weights)

        if total_weight > 0:
            # Weighted average of properties
            neighbor_props = props[neighbor_indices]
            # (K, 3) * (K, 1) -> sum -> (3,)
            weighted_props = (
                np.sum(neighbor_props * weights[:, None], axis=0) / total_weight
            )
            lce_feats[i] = weighted_props

    return lce_feats, nn_dists


def get_global_features(lattice, atoms):
    """
    Extracts global features from lattice and composition.

    Args:
        lattice (np.ndarray): (3, 3) Lattice vectors
        atoms (list): List of atomic symbols

    Returns:
        np.ndarray: (12,) vector containing:
            [a, b, c, alpha, beta, gamma, volume, density, frac_Al, frac_Ga, frac_In, total_atoms]
    """
    # 1. Lattice Lengths (a, b, c)
    if lattice.shape != (3, 3):
        return np.zeros(12)

    lengths = np.linalg.norm(lattice, axis=1)

    # 2. Lattice Angles (alpha, beta, gamma)
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]

    def get_angle(v1, v2):
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod == 0:
            return 0.0
        cos_val = np.dot(v1, v2) / norm_prod
        return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

    alpha = get_angle(b_vec, c_vec)
    beta = get_angle(a_vec, c_vec)
    gamma = get_angle(a_vec, b_vec)

    # 3. Volume
    volume = np.abs(np.linalg.det(lattice))

    # 4. Atomic Density
    n_atoms = len(atoms)
    density = n_atoms / volume if volume > 1e-6 else 0.0

    # 5. Stoichiometry (fractions of Al, Ga, In)
    counts = {"Al": 0, "Ga": 0, "In": 0}
    for s in atoms:
        if s in counts:
            counts[s] += 1

    if n_atoms > 0:
        stoich = np.array([counts["Al"], counts["Ga"], counts["In"]]) / n_atoms
    else:
        stoich = np.zeros(3)

    # 6. Total Atoms
    total_atoms_feat = float(n_atoms)

    # Combine all global features
    global_feats = np.concatenate(
        [lengths, [alpha, beta, gamma], [volume, density], stoich, [total_atoms_feat]]
    )

    return global_feats


def one_hot_encode_atoms(atoms):
    """
    One-hot encodes atom types: Al, Ga, In, O
    Returns: (N, 4) array
    """
    mapping = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    encoding = np.zeros((len(atoms), 4))
    for i, s in enumerate(atoms):
        if s in mapping:
            encoding[i, mapping[s]] = 1.0
    return encoding


def process_dataset(df, load_cached_data=True, cache_name="dataset"):
    """
    Main processing function to convert raw data into features.
    Handles caching to disk.

    Args:
        df (pd.DataFrame): Metadata dataframe
        load_cached_data (bool): Whether to try loading from cache
        cache_name (str): Filename for the cache

    Returns:
        dict: Dictionary containing numpy arrays for features and targets
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "atomic_features": data["atomic_features"],
            "global_features": data["global_features"],
            "batch_indices": data["batch_indices"],
            "targets": data["targets"],
            "ids": data["ids"],
        }

    print(f"Processing {len(df)} samples for {cache_name}...")

    all_atomic_features = []
    all_global_features = []
    batch_indices = []
    targets = []
    ids = []

    # Check if target columns exist (they won't for test set)
    has_targets = "formation_energy_ev_natom" in df.columns

    # Iterate with enumeration to create contiguous batch indices (0, 1, 2, ...)
    for i, (_, row) in enumerate(df.iterrows()):
        sample_id = row["id"]
        file_path = row["file_path"]

        # Parse geometry
        lattice, atom_symbols, coords = parse_xyz(file_path)

        # Center coordinates relative to centroid
        # Cite debug_lesson_11: Explicitly Handle Zero-Cardinality Inputs
        if len(coords) > 0:
            centroid = np.mean(coords, axis=0)
            centered_coords = coords - centroid
        else:
            centered_coords = coords

        # Compute PBC Distances
        dist_matrix = get_pbc_distances(coords, lattice)

        # 1. Atomic Features
        # One-hot encoding (4 dims)
        one_hot = one_hot_encode_atoms(atom_symbols)
        # LCE and NN Dist (3 + 1 dims)
        lce, nn_dists = get_lce_features(atom_symbols, dist_matrix)

        # Combine: [One-Hot(4), Coords(3), NN_Dist(1), LCE(3)] = 11 dims
        atomic_feats = np.concatenate([one_hot, centered_coords, nn_dists, lce], axis=1)

        # 2. Global Features (12 dims)
        global_feats = get_global_features(lattice, atom_symbols)

        # Accumulate
        all_atomic_features.append(atomic_feats)
        all_global_features.append(global_feats)

        # Batch index: map every atom in this structure to sample index 'i'
        n_atoms = len(atom_symbols)
        batch_indices.append(np.full(n_atoms, i))

        ids.append(sample_id)

        if has_targets:
            # Log transform targets: log(1 + y)
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets.append([t1, t2])
        else:
            targets.append([0.0, 0.0])  # Placeholder for test set

    # Flatten atomic features and batch indices
    # Cite debug_lesson_5: Verify Downstream Compatibility When Propagating Empty Arrays
    if all_atomic_features:
        flat_atomic = np.vstack(all_atomic_features).astype(np.float32)
    else:
        flat_atomic = np.empty((0, 11), dtype=np.float32)

    flat_batch = np.concatenate(batch_indices).astype(np.int32)

    # Global features matrix
    global_matrix = np.vstack(all_global_features).astype(np.float32)

    # Targets matrix
    targets_matrix = np.array(targets, dtype=np.float32)
    ids_array = np.array(ids, dtype=np.int32)

    # Save to cache
    np.savez(
        cache_path,
        atomic_features=flat_atomic,
        global_features=global_matrix,
        batch_indices=flat_batch,
        targets=targets_matrix,
        ids=ids_array,
    )

    return {
        "atomic_features": flat_atomic,
        "global_features": global_matrix,
        "batch_indices": flat_batch,
        "targets": targets_matrix,
        "ids": ids_array,
    }


def load_and_process_data(load_cached_data=True):
    """
    Loads metadata and processes train, val, and test sets.
    Applies standard scaling to continuous features.
    """
    train_df = pd.read_csv(METADATA_TRAIN)
    val_df = pd.read_csv(METADATA_VAL)
    test_df = pd.read_csv(METADATA_TEST)

    train_data = process_dataset(train_df, load_cached_data, "train_data")
    val_data = process_dataset(val_df, load_cached_data, "val_data")
    test_data = process_dataset(test_df, load_cached_data, "test_data")

    # Feature Scaling
    # Atomic features indices:
    # 0-3: One-hot (do not scale)
    # 4-6: Coords (scale)
    # 7: NN Dist (scale)
    # 8-10: LCE (scale)

    # Compute statistics on training data
    # Cite debug_lesson_5: Verify Downstream Compatibility
    if train_data["atomic_features"].shape[0] > 0:
        atomic_mean = np.mean(train_data["atomic_features"][:, 4:], axis=0)
        atomic_std = np.std(train_data["atomic_features"][:, 4:], axis=0) + 1e-8
    else:
        atomic_mean = np.zeros(7)
        atomic_std = np.ones(7)

    global_mean = np.mean(train_data["global_features"], axis=0)
    global_std = np.std(train_data["global_features"], axis=0) + 1e-8

    def apply_scaling(data_dict):
        # Scale atomic continuous features
        if data_dict["atomic_features"].shape[0] > 0:
            data_dict["atomic_features"][:, 4:] = (
                data_dict["atomic_features"][:, 4:] - atomic_mean
            ) / atomic_std
        # Scale global features
        data_dict["global_features"] = (
            data_dict["global_features"] - global_mean
        ) / global_std
        return data_dict

    train_data = apply_scaling(train_data)
    val_data = apply_scaling(val_data)
    test_data = apply_scaling(test_data)

    # Save scalers for reference
    np.savez(
        os.path.join(WORKING_DIR, "scalers.npz"),
        atomic_mean=atomic_mean,
        atomic_std=atomic_std,
        global_mean=global_mean,
        global_std=global_std,
    )

    return train_data, val_data, test_data
