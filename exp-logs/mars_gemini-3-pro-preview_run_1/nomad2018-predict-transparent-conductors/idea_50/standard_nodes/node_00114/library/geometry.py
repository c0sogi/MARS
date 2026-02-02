import os
import numpy as np
import pandas as pd
from library.config import Config

# Atomic properties for weighted physics features
# Values: Mass (u), Covalent Radius (Angstrom), Electronegativity (Pauling)
ATOMIC_PROPS = {
    "Al": {"mass": 26.9815, "radius": 1.21, "en": 1.61, "index": 0},
    "Ga": {"mass": 69.723, "radius": 1.22, "en": 1.81, "index": 1},
    "In": {"mass": 114.818, "radius": 1.42, "en": 1.78, "index": 2},
    "O": {"mass": 15.999, "radius": 0.66, "en": 3.44, "index": 3},
}

SPECIES_LIST = ["Al", "Ga", "In", "O"]


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.
    """
    lattice_vectors = []
    atom_types = []
    atom_coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    coords = np.array(atom_coords)
    if coords.size == 0:
        coords = coords.reshape(0, 3)
    return np.array(lattice_vectors), atom_types, coords


def get_pbc_neighbors(atom_coords, lattice_vectors, k_max):
    """
    Finds the k_max nearest neighbors for each atom under periodic boundary conditions.
    Uses a supercell approach (3x3x3) to ensure enough neighbors are found.
    """
    n_atoms = len(atom_coords)
    if n_atoms == 0:
        return np.empty((0, k_max)), np.empty((0, k_max), dtype=int)

    # Create supercell translations (-1, 0, 1)
    # 27 images including the original cell
    ranges = [-1, 0, 1]
    translations = []
    for i in ranges:
        for j in ranges:
            for k in ranges:
                translations.append(
                    i * lattice_vectors[0]
                    + j * lattice_vectors[1]
                    + k * lattice_vectors[2]
                )
    translations = np.array(translations)  # (27, 3)

    # Replicate atoms in the supercell
    # supercell_coords shape: (27 * n_atoms, 3)
    # supercell_indices maps back to original atom index (0 to n_atoms-1)
    supercell_coords = []
    supercell_indices = []

    for t in translations:
        supercell_coords.append(atom_coords + t)
        supercell_indices.append(np.arange(n_atoms))

    supercell_coords = np.vstack(supercell_coords)
    supercell_indices = np.concatenate(supercell_indices)

    # Compute distances
    # We iterate per atom to avoid massive memory usage for large N
    # For each atom in unit cell, find distances to all atoms in supercell

    all_neighbor_dists = []
    all_neighbor_indices = []

    for i in range(n_atoms):
        diff = supercell_coords - atom_coords[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))

        # Sort by distance
        sorted_args = np.argsort(dists)

        # The first one is always self (dist ~ 0), so we take 1 to k_max+1
        # Note: In a supercell, self-images might be close if the cell is small,
        # but the strictly 0 distance is the atom itself in the (0,0,0) image.
        # We want distinct atoms or images of atoms.

        # Filter out the self-interaction at distance 0 (index i in the 0,0,0 image)
        # The 0,0,0 image is usually the middle one in our generation order (index 13)
        # but simply taking > 1e-6 distance is safer.

        mask = dists > 1e-6
        valid_dists = dists[mask]
        valid_indices = supercell_indices[mask]

        # Re-sort filtered arrays
        sorted_mask_args = np.argsort(valid_dists)

        nearest_dists = valid_dists[sorted_mask_args][:k_max]
        nearest_indices = valid_indices[sorted_mask_args][:k_max]

        # Pad if not enough neighbors (unlikely with 3x3x3 supercell for K=24)
        if len(nearest_dists) < k_max:
            pad_size = k_max - len(nearest_dists)
            nearest_dists = np.pad(
                nearest_dists, (0, pad_size), "constant", constant_values=100.0
            )
            nearest_indices = np.pad(nearest_indices, (0, pad_size), "edge")

        all_neighbor_dists.append(nearest_dists)
        all_neighbor_indices.append(nearest_indices)

    return np.array(all_neighbor_dists), np.array(all_neighbor_indices)


def compute_atomic_features(atom_coords, atom_types, neighbor_dists, neighbor_indices):
    """
    Computes dense feature vector for each atom.
    """
    n_atoms = len(atom_types)

    if n_atoms == 0:
        return np.empty((0, Config.ATOMIC_FEATURE_DIM))

    # 1. One-Hot Identity (4 dims)
    one_hot = np.zeros((n_atoms, 4))
    for i, species in enumerate(atom_types):
        one_hot[i, ATOMIC_PROPS[species]["index"]] = 1.0

    # 2. Centered Coordinates (3 dims)
    centroid = np.mean(atom_coords, axis=0)
    centered_coords = atom_coords - centroid

    # 3. Nearest Neighbor Distance d_min (1 dim)
    # neighbor_dists shape (n_atoms, K_FAR)
    d_min = neighbor_dists[:, 0].reshape(-1, 1)

    # 4. Multi-Scale Packing Ratios (2 dims)
    # R_6
    d_mean_6 = np.mean(neighbor_dists[:, : Config.K_NEAR], axis=1)
    r_6 = (d_min.flatten() / (d_mean_6 + 1e-8)).reshape(-1, 1)

    # R_24
    d_mean_24 = np.mean(neighbor_dists[:, : Config.K_FAR], axis=1)
    r_24 = (d_min.flatten() / (d_mean_24 + 1e-8)).reshape(-1, 1)

    # 5. Multi-Scale Chemical Contexts (4 dims + 4 dims)
    # Helper to compute weighted context
    def get_context(k_limit):
        context = np.zeros((n_atoms, 4))
        for i in range(n_atoms):
            # Indices of neighbors for atom i
            nbs_idx = neighbor_indices[i, :k_limit]
            # Distances
            nbs_dist = neighbor_dists[i, :k_limit]

            # Inverse distance weights
            weights = 1.0 / (nbs_dist + 1e-6)

            # Sum weights per species
            for j, nb_idx in enumerate(nbs_idx):
                species = atom_types[nb_idx]
                spec_idx = ATOMIC_PROPS[species]["index"]
                context[i, spec_idx] += weights[j]

            # Normalize
            total_weight = np.sum(context[i])
            if total_weight > 0:
                context[i] /= total_weight
        return context

    context_6 = get_context(Config.K_NEAR)
    context_24 = get_context(Config.K_FAR)

    # Concatenate all features
    # 4 + 3 + 1 + 2 + 4 + 4 = 18 dims
    features = np.hstack(
        [one_hot, centered_coords, d_min, r_6, r_24, context_6, context_24]
    )

    return features


def compute_global_features(lattice_vectors, atom_types):
    """
    Computes global physics-aware features for the crystal structure.
    """
    # Lattice lengths
    a = np.linalg.norm(lattice_vectors[0])
    b = np.linalg.norm(lattice_vectors[1])
    c = np.linalg.norm(lattice_vectors[2])

    # Lattice angles (degrees)
    def angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    alpha = angle(lattice_vectors[1], lattice_vectors[2])
    beta = angle(lattice_vectors[0], lattice_vectors[2])
    gamma = angle(lattice_vectors[0], lattice_vectors[1])

    # Volume
    volume = np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )

    # Density
    n_atoms = len(atom_types)
    density = n_atoms / (volume + 1e-8)

    # Stoichiometry (4 dims)
    counts = {s: 0 for s in SPECIES_LIST}
    for s in atom_types:
        counts[s] += 1

    if n_atoms > 0:
        stoich = np.array([counts[s] for s in SPECIES_LIST]) / n_atoms
    else:
        stoich = np.zeros(4)

    # Aspect Ratios (3 dims)
    aspect_ratios = np.array([a / b, b / c, c / a])

    # Weighted Physics (3 dims: Mass, Radius, EN)
    if n_atoms > 0:
        weighted_mass = (
            sum(counts[s] * ATOMIC_PROPS[s]["mass"] for s in SPECIES_LIST) / n_atoms
        )
        weighted_radius = (
            sum(counts[s] * ATOMIC_PROPS[s]["radius"] for s in SPECIES_LIST) / n_atoms
        )
        weighted_en = (
            sum(counts[s] * ATOMIC_PROPS[s]["en"] for s in SPECIES_LIST) / n_atoms
        )
    else:
        weighted_mass = 0.0
        weighted_radius = 0.0
        weighted_en = 0.0

    # Angular Distortion (1 dim)
    ang_distortion = abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90)

    # Concatenate
    # 6 + 1 + 1 + 4 + 1 + 3 + 3 + 1 = 20 dims
    features = np.concatenate(
        [
            np.array([a, b, c, alpha, beta, gamma]),
            np.array([volume]),
            np.array([density]),
            stoich,
            np.array([n_atoms]),
            aspect_ratios,
            np.array([weighted_mass, weighted_radius, weighted_en]),
            np.array([ang_distortion]),
        ]
    )

    return features


def process_dataset(metadata_path, load_cached_data=True, cache_name="data"):
    """
    Main function to process a dataset defined by a metadata CSV.
    Handles parsing, feature computation, and caching.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return {
                "atomic_features": data["atomic_features"],
                "global_features": data["global_features"],
                "batch_indices": data["batch_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Debug mode: subset data
    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SIZE]
        print(f"Debug mode: Processing first {len(df)} samples.")

    all_atomic_features = []
    all_global_features = []
    batch_indices = []
    targets = []
    ids = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Parse
        lattice, types, coords = parse_xyz(file_path)

        # Neighbors
        dists, indices = get_pbc_neighbors(coords, lattice, Config.K_FAR)

        # Atomic Features
        af = compute_atomic_features(coords, types, dists, indices)
        all_atomic_features.append(af)

        # Global Features
        gf = compute_global_features(lattice, types)
        all_global_features.append(gf)

        # Batch Indices (maps each atom to the crystal index in this batch)
        # Using simple integer indexing 0, 1, 2... for each crystal
        n_atoms = len(types)
        batch_indices.append(
            np.full(n_atoms, idx)
        )  # Note: idx here is row index, effectively 0..N-1

        # Targets
        # Check if targets exist (train/val) or use placeholders (test)
        if "formation_energy_ev_natom" in row:
            targets.append([row["formation_energy_ev_natom"], row["bandgap_energy_ev"]])
        else:
            targets.append([0.0, 0.0])  # Placeholder

        ids.append(row["id"])

    # Concatenate
    # Atomic features: (Total_Atoms, Feature_Dim)
    # Global features: (Total_Crystals, Feature_Dim)
    # Batch indices: (Total_Atoms,) - needs to be contiguous 0,0,0,1,1,2,2...

    # Fix batch indices to be contiguous integers 0 to N-1
    # The loop above appended arrays of 'idx' which corresponds to the dataframe index.
    # If dataframe index is not 0..N-1 (e.g. after split), this might be an issue.
    # Let's re-generate batch indices strictly as 0..N-1 based on list position.

    corrected_batch_indices = []
    for i, af in enumerate(all_atomic_features):
        corrected_batch_indices.append(np.full(len(af), i))

    final_atomic = np.vstack(all_atomic_features).astype(np.float32)
    final_global = np.vstack(all_global_features).astype(np.float32)
    final_batch = np.concatenate(corrected_batch_indices).astype(np.int64)
    final_targets = np.array(targets).astype(np.float32)
    final_ids = np.array(ids).astype(np.int64)

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        atomic_features=final_atomic,
        global_features=final_global,
        batch_indices=final_batch,
        targets=final_targets,
        ids=final_ids,
    )

    return {
        "atomic_features": final_atomic,
        "global_features": final_global,
        "batch_indices": final_batch,
        "targets": final_targets,
        "ids": final_ids,
    }
