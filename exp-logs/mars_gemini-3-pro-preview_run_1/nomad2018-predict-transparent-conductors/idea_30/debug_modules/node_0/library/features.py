import os
import numpy as np
import pandas as pd
from library.config import ATOMIC_PROPERTIES, ATOM_TO_INDEX, K_NEIGHBORS, INPUT_DIR
from library.utils import compute_pbc_distances, calculate_cell_volume


def get_local_packing_density(dist_matrix, k=12):
    """
    Calculates the Local Packing Density for each atom, defined as the mean distance
    to its k nearest neighbors.

    Args:
        dist_matrix (np.ndarray): (N, N) pairwise distance matrix.
        k (int): Number of neighbors to consider.

    Returns:
        np.ndarray: (N,) array of local packing densities.
    """
    n_atoms = dist_matrix.shape[0]
    if n_atoms <= 1:
        return np.zeros(n_atoms)

    # We want k neighbors. If n_atoms - 1 < k, take all available neighbors.
    actual_k = min(k, n_atoms - 1)

    # Sort distances along each row.
    # Index 0 is self-distance (0.0), so we take indices 1 to actual_k + 1.
    sorted_dists = np.sort(dist_matrix, axis=1)

    # Slice the k nearest neighbors
    nearest_k_dists = sorted_dists[:, 1 : actual_k + 1]

    # Compute mean
    packing_density = np.mean(nearest_k_dists, axis=1)

    return packing_density


def get_weighted_global_props(atom_types):
    """
    Calculates stoichiometry-weighted mean physical properties for the crystal.
    Properties: Atomic Mass, Covalent Radius, Electronegativity.

    Args:
        atom_types (list): List of atomic symbols (e.g., ['Al', 'O', ...]).

    Returns:
        np.ndarray: (3,) array containing weighted mean mass, radius, and electronegativity.
    """
    if not atom_types:
        return np.zeros(3)

    total_atoms = len(atom_types)

    total_mass = 0.0
    total_radius = 0.0
    total_electronegativity = 0.0

    for atom in atom_types:
        props = ATOMIC_PROPERTIES.get(atom, {})
        total_mass += props.get("mass", 0.0)
        total_radius += props.get("radius", 0.0)
        total_electronegativity += props.get("electronegativity", 0.0)

    # Calculate means
    mean_mass = total_mass / total_atoms
    mean_radius = total_radius / total_atoms
    mean_electronegativity = total_electronegativity / total_atoms

    return np.array([mean_mass, mean_radius, mean_electronegativity])


def parse_xyz_file(file_path):
    """
    Parses the custom XYZ format provided in the dataset.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple: (lattice_matrix, atom_types, atom_coords)
    """
    lattice_vectors = []
    atom_types = []
    atom_coords = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z symbol
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(atom_coords)


def process_geometry(file_path):
    """
    Extracts atomic and global features from a single geometry file.

    Args:
        file_path (str): Path to the geometry file.

    Returns:
        tuple: (atomic_features, global_features)
            atomic_features: (N, 9) array
            global_features: (15,) array
    """
    # 1. Parse File
    lattice_matrix, atom_types, coords = parse_xyz_file(file_path)
    n_atoms = len(atom_types)

    # ---------------------------
    # Global Features Construction
    # ---------------------------

    # Lattice parameters (lengths and angles)
    # Lengths
    a = np.linalg.norm(lattice_matrix[0])
    b = np.linalg.norm(lattice_matrix[1])
    c = np.linalg.norm(lattice_matrix[2])

    # Angles (in degrees)
    alpha = np.degrees(
        np.arccos(np.dot(lattice_matrix[1], lattice_matrix[2]) / (b * c))
    )
    beta = np.degrees(np.arccos(np.dot(lattice_matrix[0], lattice_matrix[2]) / (a * c)))
    gamma = np.degrees(
        np.arccos(np.dot(lattice_matrix[0], lattice_matrix[1]) / (a * b))
    )

    # Volume and Density
    volume = calculate_cell_volume(lattice_matrix)
    density = n_atoms / volume if volume > 0 else 0.0

    # Stoichiometry (Al, Ga, In fractions relative to total atoms)
    # Note: We compute fractions of Al, Ga, In. Oxygen is implicitly handled by the remaining fraction
    # or by the physics properties. The config specifies 3 stoichiometry dims.
    counts = {"Al": 0, "Ga": 0, "In": 0, "O": 0}
    for at in atom_types:
        if at in counts:
            counts[at] += 1

    frac_al = counts["Al"] / n_atoms
    frac_ga = counts["Ga"] / n_atoms
    frac_in = counts["In"] / n_atoms

    # Weighted Physical Properties
    weighted_props = get_weighted_global_props(atom_types)

    # Assemble Global Vector (15 dims)
    # [a, b, c, alpha, beta, gamma, vol, density, n_atoms, frac_al, frac_ga, frac_in, mass, radius, eneg]
    global_feats = np.array(
        [
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            volume,
            density,
            float(n_atoms),
            frac_al,
            frac_ga,
            frac_in,
            weighted_props[0],
            weighted_props[1],
            weighted_props[2],
        ]
    )

    # ---------------------------
    # Atomic Features Construction
    # ---------------------------

    # 1. One-Hot Encoding (4 dims)
    one_hot = np.zeros((n_atoms, 4))
    for i, at in enumerate(atom_types):
        if at in ATOM_TO_INDEX:
            one_hot[i, ATOM_TO_INDEX[at]] = 1.0

    # 2. Centered Coordinates (3 dims)
    # Center relative to centroid
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 3. Geometric Descriptors (PBC Distance based)
    dist_matrix = compute_pbc_distances(coords, lattice_matrix)

    # Nearest Neighbor Distance (1 dim)
    # Fill diagonal with infinity to ignore self-distance
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dist = np.min(dist_matrix, axis=1).reshape(-1, 1)

    # Local Packing Density (1 dim)
    # Restore diagonal to 0 for consistent sorting logic in helper, or handle in helper
    # The helper expects actual distances. Let's pass the matrix with 0 on diagonal.
    np.fill_diagonal(dist_matrix, 0.0)
    packing_density = get_local_packing_density(dist_matrix, k=K_NEIGHBORS).reshape(
        -1, 1
    )

    # Assemble Atomic Vector (9 dims)
    # [OneHot(4), Coords(3), NN(1), Density(1)]
    atomic_feats = np.hstack([one_hot, centered_coords, nn_dist, packing_density])

    return atomic_feats, global_feats


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Processes the dataset defined by the metadata file.
    Handles caching to avoid re-processing.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the .npz cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            - 'atomic_features_flat': Concatenated atomic features (Total_Atoms, 9)
            - 'atom_counts': Number of atoms per sample (N_samples,)
            - 'global_features': Global features (N_samples, 15)
            - 'targets': Target values (N_samples, 2) (if available, else zeros)
            - 'ids': Sample IDs (N_samples,)
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_atomic_feats = []
    all_global_feats = []
    all_targets = []
    all_ids = []
    atom_counts = []

    # Check if targets exist (train/val vs test)
    has_targets = (
        "formation_energy_ev_natom" in df.columns and "bandgap_energy_ev" in df.columns
    )

    for _, row in df.iterrows():
        sample_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}. Skipping.")
            continue

        # Extract features
        atomic_f, global_f = process_geometry(full_path)

        all_atomic_feats.append(atomic_f)
        all_global_feats.append(global_f)
        atom_counts.append(len(atomic_f))
        all_ids.append(sample_id)

        if has_targets:
            all_targets.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )
        else:
            all_targets.append([0.0, 0.0])  # Placeholder for test set

    # Concatenate atomic features into a single flat array
    # This avoids pickle issues and allows efficient storage
    atomic_features_flat = np.vstack(all_atomic_feats).astype(np.float32)
    global_features = np.vstack(all_global_feats).astype(np.float32)
    targets = np.vstack(all_targets).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int32)
    atom_counts = np.array(atom_counts, dtype=np.int32)

    # 3. Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        atomic_features_flat=atomic_features_flat,
        atom_counts=atom_counts,
        global_features=global_features,
        targets=targets,
        ids=ids,
    )

    return {
        "atomic_features_flat": atomic_features_flat,
        "atom_counts": atom_counts,
        "global_features": global_features,
        "targets": targets,
        "ids": ids,
    }
