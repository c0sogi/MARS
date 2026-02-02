import os
import numpy as np
import pandas as pd
from library.config import Config


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file.

    Args:
        file_path (str): Path to the xyz file.

    Returns:
        lattice (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols.
        coords (np.ndarray): N_atoms x 3 array of atomic coordinates.
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
                lattice_vectors.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif parts[0] == "atom":
                # format: atom x y z type
                atom_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(atom_coords)


def compute_lattice_features(lattice):
    """
    Computes macroscopic lattice features.

    Args:
        lattice (np.ndarray): 3x3 lattice matrix.

    Returns:
        lengths (np.ndarray): [a, b, c]
        angles (np.ndarray): [alpha, beta, gamma] in degrees
        volume (float): Unit cell volume
    """
    # Lattice vectors
    a = lattice[0]
    b = lattice[1]
    c = lattice[2]

    # Lengths
    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)
    len_c = np.linalg.norm(c)
    lengths = np.array([len_a, len_b, len_c])

    # Angles
    # alpha between b and c
    alpha_rad = np.arccos(np.clip(np.dot(b, c) / (len_b * len_c), -1.0, 1.0))
    # beta between a and c
    beta_rad = np.arccos(np.clip(np.dot(a, c) / (len_a * len_c), -1.0, 1.0))
    # gamma between a and b
    gamma_rad = np.arccos(np.clip(np.dot(a, b) / (len_a * len_b), -1.0, 1.0))

    angles = np.degrees(np.array([alpha_rad, beta_rad, gamma_rad]))

    # Volume: scalar triple product
    volume = np.abs(np.dot(a, np.cross(b, c)))

    return lengths, angles, volume


def compute_pbc_shells(coords, lattice):
    """
    Computes Bonding Shell and Coordination Shell distances respecting PBC.

    Args:
        coords (np.ndarray): N x 3 atomic coordinates.
        lattice (np.ndarray): 3x3 lattice vectors.

    Returns:
        bonding_dists (np.ndarray): N x 1 array of nearest neighbor distances.
        coord_dists (np.ndarray): N x 1 array of mean distances to 2nd-5th neighbors.
    """
    n_atoms = len(coords)

    # Generate image offsets (3x3x3 grid)
    # Range -1 to 1
    ranges = [0, -1, 1]
    shifts = []
    for i in ranges:
        for j in ranges:
            for k in ranges:
                shifts.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
    shifts = np.array(shifts)  # 27 x 3

    # Replicate coords for all images
    # We want to compute distance from each atom in the central cell (original coords)
    # to all atoms in all image cells.

    all_image_coords = []
    for shift in shifts:
        all_image_coords.append(coords + shift)
    all_image_coords = np.vstack(all_image_coords)  # (27*N, 3)

    # Compute distances
    # shape: (N, 27*N)
    # We use simple numpy broadcasting.
    # For N ~ 80, 27*N ~ 2160. Matrix is 80x2160.

    dists_sq = np.sum(
        (coords[:, np.newaxis, :] - all_image_coords[np.newaxis, :, :]) ** 2, axis=2
    )
    dists = np.sqrt(dists_sq)

    # Sort distances for each atom
    dists.sort(axis=1)

    # Column 0 is always 0.0 (distance to self in central image)
    # We slice off the self-distance first.
    neighbor_dists = dists[:, 1:]

    # Bonding Shell: 1st nearest neighbor (index 0 in sliced array)
    bonding_dists = neighbor_dists[:, 0]

    # Coordination Shell: Mean of neighbors defined in Config
    # Config defines start=1 (2nd NN) and end=5 (up to 5th NN)
    # In python slice [1:5] gives indices 1, 2, 3, 4.
    start = Config.COORD_SHELL_START_IDX
    end = Config.COORD_SHELL_END_IDX
    coord_dists = np.mean(neighbor_dists[:, start:end], axis=1)

    return bonding_dists, coord_dists


def process_geometry_file(file_path):
    """
    Process a single geometry file to extract atomic and global features.
    """
    # 1. Parse
    lattice, atom_types, coords = parse_xyz(file_path)

    # 2. Centering
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 3. PBC Shells
    bonding, coord_shells = compute_pbc_shells(coords, lattice)

    # 4. Global Features
    lengths, angles, volume = compute_lattice_features(lattice)
    num_atoms = len(atom_types)
    density = num_atoms / volume

    # Stoichiometry
    # Counts
    counts = {"Al": 0, "Ga": 0, "In": 0, "O": 0}
    for at in atom_types:
        if at in counts:
            counts[at] += 1

    # Fractions relative to total atoms
    frac_Al = counts["Al"] / num_atoms
    frac_Ga = counts["Ga"] / num_atoms
    frac_In = counts["In"] / num_atoms

    global_feats = np.array(
        [
            lengths[0],
            lengths[1],
            lengths[2],
            angles[0],
            angles[1],
            angles[2],
            volume,
            density,
            frac_Al,
            frac_Ga,
            frac_In,
            float(num_atoms),
        ]
    )

    # Atomic Features
    # One-hot encoding: Al, Ga, In, O
    # Map: Al: [1,0,0,0], Ga: [0,1,0,0], In: [0,0,1,0], O: [0,0,0,1]
    type_map = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    one_hots = np.zeros((num_atoms, 4))
    for i, at in enumerate(atom_types):
        if at in type_map:
            one_hots[i, type_map[at]] = 1.0

    # Concatenate atomic features: [One-hot(4), Centered(3), Bond(1), Coord(1)]
    # Total 9 dims
    atomic_feats = np.hstack(
        [one_hots, centered_coords, bonding.reshape(-1, 1), coord_shells.reshape(-1, 1)]
    )

    return atomic_feats, global_feats


def process_dataset(metadata_path, output_path, load_cached_data=True):
    """
    Main function to process a dataset (train/val/test).
    Handles caching logic.
    """
    # Directory safety
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached data from {output_path}...")
        try:
            data = np.load(output_path)
            return {
                "node_feats": data["node_feats"],
                "global_feats": data["global_feats"],
                "batch_indices": data["batch_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    print(f"Processing dataset from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_node_feats = []
    all_global_feats = []
    all_batch_indices = []
    all_targets = []
    all_ids = []

    input_dir = Config.INPUT_DIR

    for idx, row in df.iterrows():
        # Construct full file path
        # metadata file_path is relative to input dir, e.g. "train/1/geometry.xyz"
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        if not os.path.exists(full_path):
            continue

        # Process geometry
        atomic_feats, global_feats = process_geometry_file(full_path)

        n_atoms = atomic_feats.shape[0]

        # Accumulate
        all_node_feats.append(atomic_feats)
        all_global_feats.append(global_feats)

        # Batch index: assigns each atom to the sample index (0 to N_samples-1)
        sample_idx = len(all_ids)
        all_batch_indices.append(np.full(n_atoms, sample_idx))

        all_ids.append(row["id"])

        # Targets
        if "formation_energy_ev_natom" in row:
            t1 = row["formation_energy_ev_natom"]
            t2 = row["bandgap_energy_ev"]
            all_targets.append([t1, t2])
        else:
            # Placeholder for test
            all_targets.append([0.0, 0.0])

    # Concatenate
    node_feats_arr = np.vstack(all_node_feats).astype(np.float32)
    global_feats_arr = np.vstack(all_global_feats).astype(np.float32)
    batch_indices_arr = np.concatenate(all_batch_indices).astype(np.int64)
    targets_arr = np.array(all_targets).astype(np.float32)
    ids_arr = np.array(all_ids).astype(np.int64)

    # 3. Save to Cache
    print(f"Saving processed data to {output_path}...")
    np.savez(
        output_path,
        node_feats=node_feats_arr,
        global_feats=global_feats_arr,
        batch_indices=batch_indices_arr,
        targets=targets_arr,
        ids=ids_arr,
    )

    return {
        "node_feats": node_feats_arr,
        "global_feats": global_feats_arr,
        "batch_indices": batch_indices_arr,
        "targets": targets_arr,
        "ids": ids_arr,
    }
