import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic species, and coordinates.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Lines 2-4 contain lattice vectors (ASE format usually)
    # Format: lattice_vector x y z
    lattice_vectors = []
    for line in lines:
        parts = line.strip().split()
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])

    lattice_vectors = np.array(lattice_vectors)

    # Remaining lines contain atoms
    # Format: atom x y z symbol
    atom_types = []
    coords = []
    for line in lines:
        parts = line.strip().split()
        if parts[0] == "atom":
            coords.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    return np.array(atom_types), np.array(coords), lattice_vectors


def get_pbc_neighbors(coords, lattice_vectors, k_max):
    """
    Finds neighbors considering PBC by expanding the unit cell.
    Returns sorted distances and indices of neighbors for each atom.
    """
    n_atoms = len(coords)

    # Create supercell (3x3x3)
    # Offsets in fractional coordinates
    ranges = [-1, 0, 1]
    offsets = np.array([[i, j, k] for i in ranges for j in ranges for k in ranges])

    # Convert offsets to Cartesian
    cart_offsets = offsets @ lattice_vectors

    all_coords = []
    all_indices = []

    for i in range(len(cart_offsets)):
        offset = cart_offsets[i]
        all_coords.append(coords + offset)
        all_indices.append(np.arange(n_atoms))

    all_coords = np.vstack(all_coords)
    all_indices = np.concatenate(all_indices)

    # Compute distances from original atoms to all atoms in supercell
    # Shape: (n_atoms, n_atoms * 27)
    # Using broadcasting
    # coords: (N, 3) -> (N, 1, 3)
    # all_coords: (M, 3) -> (1, M, 3)
    dists = np.linalg.norm(coords[:, None, :] - all_coords[None, :, :], axis=2)

    # Sort distances
    sorted_indices = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, sorted_indices, axis=1)

    # Map back to original atom indices
    neighbor_indices = all_indices[sorted_indices]

    # Exclude self (first column is always dist 0 to self)
    # We take k_max neighbors
    # Note: The first neighbor is the atom itself (dist=0), so we take 1:k_max+1

    return sorted_dists[:, 1 : k_max + 1], neighbor_indices[:, 1 : k_max + 1]


def get_local_features(atom_types, coords, lattice_vectors):
    """
    Computes atomic features:
    - One-hot ID
    - Centered Coords
    - NN Dist
    - Packing Ratio
    - Chemical Contexts (K=6, K=24)
    """
    n_atoms = len(atom_types)

    # 1. Atomic Identity (One-hot)
    one_hot = np.zeros((n_atoms, Config.NUM_ATOM_TYPES))
    for i, atom in enumerate(atom_types):
        one_hot[i, Config.ATOM_TO_IDX[atom]] = 1.0

    # 2. Spatial Context (Centered Coords)
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # Neighbors
    k_max = max(Config.K_NEAR, Config.K_FAR, 12)  # Need 12 for packing ratio
    dists, indices = get_pbc_neighbors(coords, lattice_vectors, k_max)

    # 3. Nearest Neighbor Distance
    nn_dist = dists[:, 0:1]  # Shape (N, 1)

    # 4. Local Packing Ratio
    # Mean distance to 12 nearest neighbors
    mean_12 = np.mean(dists[:, :12], axis=1, keepdims=True)
    packing_ratio = nn_dist / (mean_12 + 1e-8)

    # 5. Multi-Scale Chemical Contexts
    def compute_context(k):
        k_dists = dists[:, :k]
        k_indices = indices[:, :k]

        # Inverse distance weights
        weights = 1.0 / (k_dists + 1e-6)

        # Gather neighbor types
        # neighbor_types_idx shape: (N, k)
        neighbor_types_str = atom_types[k_indices]

        # Convert to one-hot: (N, k, 4)
        n_one_hot = np.zeros((n_atoms, k, Config.NUM_ATOM_TYPES))
        for i in range(n_atoms):
            for j in range(k):
                n_one_hot[i, j, Config.ATOM_TO_IDX[neighbor_types_str[i, j]]] = 1.0

        # Weighted sum: (N, 4)
        weighted_sum = np.sum(n_one_hot * weights[:, :, None], axis=1)

        # Normalize by sum of weights
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        context = weighted_sum / (sum_weights + 1e-8)
        return context

    ctx_near = compute_context(Config.K_NEAR)
    ctx_far = compute_context(Config.K_FAR)

    # Concatenate all atomic features
    # 4 + 3 + 1 + 1 + 4 + 4 = 17 dimensions
    atomic_features = np.hstack(
        [one_hot, centered_coords, nn_dist, packing_ratio, ctx_near, ctx_far]
    )

    return atomic_features


def get_global_features(atom_types, coords, lattice_vectors):
    """
    Computes global features:
    - Lattice params
    - Volume, Density
    - Stoichiometry
    - N_atoms
    - Aspect Ratios
    - Weighted Physics
    - Global Bond Statistics
    """
    # 1. Lattice Parameters
    a = np.linalg.norm(lattice_vectors[0])
    b = np.linalg.norm(lattice_vectors[1])
    c = np.linalg.norm(lattice_vectors[2])

    alpha = np.degrees(
        np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c))
    )
    beta = np.degrees(
        np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c))
    )
    gamma = np.degrees(
        np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b))
    )

    lattice_feats = np.array([a, b, c, alpha, beta, gamma])

    # 2. Volume & Density
    # Scalar triple product
    volume = np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )
    n_atoms = len(atom_types)
    density = n_atoms / volume

    # 3. Stoichiometry & N_atoms
    counts = {el: 0 for el in Config.ATOM_LIST}
    for at in atom_types:
        counts[at] += 1

    stoich = np.array([counts[el] for el in Config.ATOM_LIST]) / n_atoms

    # 4. Aspect Ratios
    aspect_ratios = np.array([a / b, b / c, c / a])

    # 5. Weighted Physics
    mass = sum(Config.ATOMIC_MASS[at] for at in atom_types) / n_atoms
    radius = sum(Config.ATOMIC_RADIUS[at] for at in atom_types) / n_atoms
    eneg = sum(Config.ELECTRONEGATIVITY[at] for at in atom_types) / n_atoms
    physics = np.array([mass, radius, eneg])

    # 6. Global Bond Statistics
    # Compute pairwise distances in unit cell using MIC
    # We iterate over all unique pairs (i, j) with i < j
    # For each pair type, we average the distances

    # Calculate MIC distances for all pairs in unit cell
    # r_ij = r_i - r_j
    # fractional
    inv_lattice = np.linalg.inv(lattice_vectors.T)
    frac_coords = coords @ inv_lattice

    bond_sums = {pair: 0.0 for pair in Config.BOND_PAIRS}
    bond_counts = {pair: 0 for pair in Config.BOND_PAIRS}

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            diff = frac_coords[i] - frac_coords[j]
            diff = diff - np.round(diff)
            cart_diff = diff @ lattice_vectors
            dist = np.linalg.norm(cart_diff)

            # Identify pair type
            t1 = atom_types[i]
            t2 = atom_types[j]
            # Sort to match BOND_PAIRS keys
            pair_key = tuple(sorted((t1, t2)))

            if pair_key in bond_sums:
                bond_sums[pair_key] += dist
                bond_counts[pair_key] += 1

    bond_stats = []
    for pair in Config.BOND_PAIRS:
        if bond_counts[pair] > 0:
            bond_stats.append(bond_sums[pair] / bond_counts[pair])
        else:
            bond_stats.append(0.0)
    bond_stats = np.array(bond_stats)

    # Concatenate
    # 6 + 1 + 1 + 4 + 1 + 3 + 3 + 10 = 29
    global_features = np.concatenate(
        [
            lattice_feats,
            [volume, density],
            stoich,
            [n_atoms],
            aspect_ratios,
            physics,
            bond_stats,
        ]
    )

    return global_features


def process_data(metadata_path, cache_file, load_cached_data=True):
    """
    Main processing function.
    Reads metadata, parses XYZ files, extracts features, and returns arrays.
    """

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file)
        return (
            data["atomic_features"],
            data["global_features"],
            data["targets"],
            data["ids"],
            data["batch_indices"],
        )

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_atomic_feats = []
    all_global_feats = []
    all_targets = []
    all_ids = []
    batch_indices = []  # To map atoms to graphs

    # Check if targets exist (test set might not have them)
    has_targets = "formation_energy_ev_natom" in df.columns

    for idx, row in df.iterrows():
        # Construct full file path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} not found. Skipping.")
            continue

        atom_types, coords, lattice_vectors = parse_xyz(full_path)

        # Extract features
        af = get_local_features(atom_types, coords, lattice_vectors)
        gf = get_global_features(atom_types, coords, lattice_vectors)

        # Collect
        all_atomic_feats.append(af)
        all_global_feats.append(gf)
        all_ids.append(row["id"])

        # Batch index for this graph (all atoms in this crystal get this graph_idx)
        # We will flatten atomic feats later, so we need this index to pool back
        n_atoms = len(atom_types)
        batch_indices.append(np.full(n_atoms, idx))

        if has_targets:
            all_targets.append(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            )

    # Concatenate
    # Atomic features: (Total_Atoms, ATOMIC_FEATURE_DIM)
    atomic_features_stacked = np.vstack(all_atomic_feats)

    # Global features: (N_samples, GLOBAL_FEATURE_DIM)
    global_features_stacked = np.vstack(all_global_feats)

    # Batch indices: (Total_Atoms,)
    # Note: This is 0,0,0... 1,1... corresponding to sample index in global_features
    # However, since we might skip files, we should re-index to be contiguous 0..N-1
    # But here we append sequentially, so it is contiguous 0, 1, 2...
    batch_indices_stacked = np.concatenate(batch_indices)

    if has_targets:
        targets_stacked = np.array(all_targets)
    else:
        targets_stacked = np.zeros((len(all_ids), 2))  # Placeholder

    ids_stacked = np.array(all_ids)

    # Save to cache
    np.savez(
        cache_file,
        atomic_features=atomic_features_stacked,
        global_features=global_features_stacked,
        targets=targets_stacked,
        ids=ids_stacked,
        batch_indices=batch_indices_stacked,
    )

    print(f"Data processed and saved to {cache_file}")

    return (
        atomic_features_stacked,
        global_features_stacked,
        targets_stacked,
        ids_stacked,
        batch_indices_stacked,
    )
