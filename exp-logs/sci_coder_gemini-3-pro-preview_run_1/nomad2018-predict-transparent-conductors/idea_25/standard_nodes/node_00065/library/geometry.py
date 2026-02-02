import os
import numpy as np
import torch
import pandas as pd
from library.config import (
    ATOM_TYPES,
    LARGE_DISTANCE_CONSTANT,
    INPUT_DIR,
    ATOMIC_INPUT_DIM,
    GLOBAL_INPUT_DIM,
)


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors and atomic information.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple: (lattice_matrix, atom_types, atom_coords)
            - lattice_matrix: np.array of shape (3, 3)
            - atom_types: list of strings (element symbols)
            - atom_coords: np.array of shape (N, 3)
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
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(atom_coords)


def compute_lattice_properties(lattice):
    """
    Computes global lattice properties: lengths, angles, and volume.

    Args:
        lattice (np.array): 3x3 lattice matrix.

    Returns:
        tuple: (lengths, angles_deg, volume)
            - lengths: np.array [a, b, c]
            - angles_deg: np.array [alpha, beta, gamma]
            - volume: float
    """
    # Lattice vector lengths
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])
    lengths = np.array([a, b, c])

    # Volume: scalar triple product
    volume = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))

    # Angles
    # alpha: angle between b and c
    # beta: angle between a and c
    # gamma: angle between a and b
    alpha_rad = np.arccos(np.clip(np.dot(lattice[1], lattice[2]) / (b * c), -1.0, 1.0))
    beta_rad = np.arccos(np.clip(np.dot(lattice[0], lattice[2]) / (a * c), -1.0, 1.0))
    gamma_rad = np.arccos(np.clip(np.dot(lattice[0], lattice[1]) / (a * b), -1.0, 1.0))

    angles_deg = np.degrees([alpha_rad, beta_rad, gamma_rad])

    return lengths, angles_deg, volume


def get_pbc_shift_vectors(lattice):
    """
    Generates the 27 shift vectors for periodic boundary conditions (neighboring cells).
    """
    shifts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                shifts.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
    return np.array(shifts)


def compute_chemically_split_features(atom_types, coords, lattice):
    """
    Computes atomic features including one-hot encoding, centered coords,
    and chemically split nearest neighbor distances (homo/hetero).

    Args:
        atom_types (list): List of element symbols.
        coords (np.array): (N, 3) atomic coordinates.
        lattice (np.array): 3x3 lattice matrix.

    Returns:
        np.array: (N, 9) feature matrix.
    """
    n_atoms = len(atom_types)

    # 1. One-hot encoding
    type_indices = [ATOM_TYPES.index(t) for t in atom_types]
    one_hot = np.zeros((n_atoms, len(ATOM_TYPES)))
    one_hot[np.arange(n_atoms), type_indices] = 1.0

    # 2. Centered coordinates
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 3. PBC Distances
    # Generate 27 images of the unit cell
    shift_vectors = get_pbc_shift_vectors(lattice)  # (27, 3)

    # Replicate coords for all shifts: (N, 27, 3)
    # coords[:, None, :] shape is (N, 1, 3)
    # shift_vectors[None, :, :] shape is (1, 27, 3)
    # all_image_coords shape: (N, 27, 3) -> reshape to (N*27, 3)
    all_image_coords = (coords[:, None, :] + shift_vectors[None, :, :]).reshape(-1, 3)

    # Replicate types for all images
    all_image_types = np.array(atom_types * 27)  # list repetition

    # Compute pairwise distances between original atoms (N) and all image atoms (N*27)
    # We use broadcasting or simple loop if N is small. N ~ 10-80.
    # dist_matrix: (N, N*27)
    # Using squared euclidean first to avoid sqrt on zeros if any
    dists = np.linalg.norm(coords[:, None, :] - all_image_coords[None, :, :], axis=2)

    homo_dists = []
    hetero_dists = []

    for i in range(n_atoms):
        current_type = atom_types[i]

        # Masks for image atoms
        is_homo = all_image_types == current_type
        is_hetero = all_image_types != current_type

        # Distances to homo atoms
        d_homo_all = dists[i][is_homo]
        # Filter out self-distance (which is 0.0)
        # We assume any distance < 1e-4 is self-interaction of the same atom at (0,0,0) shift
        valid_homo = d_homo_all[d_homo_all > 1e-4]

        if len(valid_homo) > 0:
            homo_dists.append(np.min(valid_homo))
        else:
            # Should theoretically not happen in a periodic crystal unless N=1 and large cell?
            # Even for N=1, periodic images exist.
            homo_dists.append(LARGE_DISTANCE_CONSTANT)

        # Distances to hetero atoms
        d_hetero_all = dists[i][is_hetero]
        if len(d_hetero_all) > 0:
            hetero_dists.append(np.min(d_hetero_all))
        else:
            # Pure element crystal
            hetero_dists.append(LARGE_DISTANCE_CONSTANT)

    homo_dists = np.array(homo_dists).reshape(-1, 1)
    hetero_dists = np.array(hetero_dists).reshape(-1, 1)

    # Concatenate all features: One-hot (4) + Coords (3) + Homo (1) + Hetero (1) = 9
    features = np.hstack([one_hot, centered_coords, homo_dists, hetero_dists])

    return features.astype(np.float32)


def process_file(file_path):
    """
    Process a single geometry file to extract atomic and global features.

    Args:
        file_path (str): Relative path to geometry.xyz

    Returns:
        tuple: (atomic_features, global_features)
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    lattice, types, coords = parse_xyz(full_path)

    # Atomic Features
    atomic_feats = compute_chemically_split_features(types, coords, lattice)

    # Global Features
    lengths, angles, volume = compute_lattice_properties(lattice)
    n_atoms = len(types)
    density = n_atoms / volume

    # Composition (Al, Ga, In)
    # ATOM_TYPES = ["Al", "Ga", "In", "O"]
    # We want fractions of Al, Ga, In specifically as per prompt description
    # "Relative compositions of Al, Ga, and In"
    # We calculate fraction relative to total atoms.
    comp = [types.count(el) / n_atoms for el in ["Al", "Ga", "In"]]

    # Construct global vector:
    # [len_a, len_b, len_c, alpha, beta, gamma, vol, dens, n_atoms, frac_Al, frac_Ga, frac_In]
    # Size: 3 + 3 + 1 + 1 + 1 + 3 = 12
    global_feats = np.concatenate(
        [lengths, angles, [volume, density, float(n_atoms)], comp]
    )

    return atomic_feats, global_feats.astype(np.float32)


def load_and_process_data(metadata_df, cache_path, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes raw files and caches the result.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' and targets.
        cache_path (str): Path to save/load the processed .pt file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed tensors.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return torch.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {len(metadata_df)} samples...")

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    # Check if targets exist (they might not for test set)
    has_targets = "formation_energy_ev_natom" in metadata_df.columns

    for idx, row in metadata_df.iterrows():
        # Process geometry
        af, gf = process_file(row["file_path"])

        atomic_features_list.append(torch.tensor(af, dtype=torch.float32))
        global_features_list.append(torch.tensor(gf, dtype=torch.float32))
        ids_list.append(row["id"])

        if has_targets:
            # Targets: formation_energy, bandgap_energy
            # Apply log(1+x) transformation to targets as per strategy
            # Note: targets can be negative? Formation energy usually negative/positive.
            # Bandgap is positive.
            # The prompt metric is RMSLE, which implies targets are positive or we care about log scale.
            # However, formation energy can be negative.
            # Standard RMSLE is usually for positive values.
            # Let's check data analysis output: Formation energy min=0.0, max=0.6572. Bandgap min=0.0057.
            # Both are non-negative. Safe to use log1p.
            t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            targets_list.append(torch.tensor(t, dtype=torch.float32))

    # Stack global features (fixed size)
    global_features_tensor = torch.stack(global_features_list)

    # Targets tensor
    if has_targets:
        targets_tensor = torch.stack(targets_list)
    else:
        targets_tensor = torch.zeros((len(ids_list), 2))  # Placeholder

    data = {
        "atomic_features": atomic_features_list,  # List of tensors (variable length)
        "global_features": global_features_tensor,
        "targets": targets_tensor,
        "ids": ids_list,
    }

    print(f"Saving processed data to {cache_path}...")
    torch.save(data, cache_path)

    return data
