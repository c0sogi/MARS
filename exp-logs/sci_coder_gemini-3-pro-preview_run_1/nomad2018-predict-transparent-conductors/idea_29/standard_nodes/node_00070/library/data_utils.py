import os
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    ATOMIC_PROPERTIES,
    ATOM_TO_INDEX,
    NUM_ATOM_TYPES,
    EXCLUDE_COLS,
    TARGET_COLS,
)


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic coordinates, and atom types.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols (str).
        coords (np.ndarray): Nx3 array of atomic Cartesian coordinates.
    """
    lattice_vectors = []
    atom_types = []
    coords = []

    full_path = os.path.join(INPUT_DIR, file_path)

    with open(full_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(coords)


def get_pbc_distances(coords, lattice):
    """
    Calculates the distance to the nearest neighbor for each atom, respecting
    periodic boundary conditions (PBC).

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        nn_dists (np.ndarray): (N,) array of distances to the nearest neighbor.
    """
    n_atoms = coords.shape[0]

    # If there's only one atom (unlikely in crystals), return a large distance or 0
    if n_atoms < 2:
        return np.zeros(n_atoms)

    # Generate 27 periodic images (3x3x3 grid centered at 0)
    # Shifts indices: -1, 0, 1 for each dimension
    shifts = np.array(
        [
            i * lattice[0] + j * lattice[1] + k * lattice[2]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
        ]
    )  # Shape: (27, 3)

    # Compute pairwise difference vectors: diff[i, j] = coords[i] - coords[j]
    # Shape: (N, N, 3)
    diff_vectors = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # We need to find min distance |d_ij + shift| over all shifts
    # Expand diff_vectors to (1, N, N, 3) and shifts to (27, 1, 1, 3)
    # Result: (27, N, N, 3)
    all_diffs = diff_vectors[np.newaxis, :, :, :] + shifts[:, np.newaxis, np.newaxis, :]

    # Compute squared distances: (27, N, N)
    dists_sq = np.sum(all_diffs**2, axis=-1)

    # Find minimum distance across all images for each pair: (N, N)
    min_dists_sq = np.min(dists_sq, axis=0)

    # Mask diagonal (self-distance is 0) with infinity to find nearest neighbor
    np.fill_diagonal(min_dists_sq, np.inf)

    # Nearest neighbor distance for each atom
    nn_dists = np.sqrt(np.min(min_dists_sq, axis=1))

    return nn_dists


def center_coordinates(coords):
    """
    Centers atomic coordinates relative to their centroid.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.

    Returns:
        centered_coords (np.ndarray): Nx3 array.
    """
    centroid = np.mean(coords, axis=0)
    return coords - centroid


def calculate_physics_features(atom_types):
    """
    Computes stoichiometry-weighted average physical properties.

    Args:
        atom_types (list): List of atomic symbols.

    Returns:
        features (np.ndarray): Array of [avg_mass, avg_radius, avg_electronegativity].
    """
    n_atoms = len(atom_types)
    if n_atoms == 0:
        return np.zeros(3)

    mass_sum = 0.0
    radius_sum = 0.0
    en_sum = 0.0

    for atom in atom_types:
        props = ATOMIC_PROPERTIES.get(atom, [0.0, 0.0, 0.0])
        mass_sum += props[0]
        radius_sum += props[1]
        en_sum += props[2]

    return np.array([mass_sum / n_atoms, radius_sum / n_atoms, en_sum / n_atoms])


def process_dataset(metadata_path, load_cached_data=True):
    """
    Main data processing function. Loads metadata, processes geometry and tabular features,
    and returns structured data. Implements caching.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        data_dict (dict): Dictionary containing:
            - 'atomic_features': List of np.arrays (one per crystal) for local stream.
            - 'global_features': np.ndarray (N_samples, Global_Dim).
            - 'targets': np.ndarray (N_samples, 2) (NaN for test set).
            - 'ids': np.ndarray (N_samples,).
    """
    # Determine cache file name based on metadata filename
    base_name = os.path.basename(metadata_path).replace(".csv", "")
    cache_path = os.path.join(WORKING_DIR, f"{base_name}_data.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # atomic_features is saved as an object array of arrays, need to convert back to list
            return {
                "atomic_features": list(loaded["atomic_features"]),
                "global_features": loaded["global_features"],
                "targets": loaded["targets"],
                "ids": loaded["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    # Pre-identify columns for global features from CSV
    # We use: lattice vectors/angles, volume (if calc), density (if calc), composition
    # Note: We will recalculate volume/density to be consistent with XYZ if needed,
    # but using CSV provided columns is safer for consistency with "Tabular Preprocessing".
    # The prompt asks to calculate Volume and Density.

    for idx, row in df.iterrows():
        # --- Geometry Processing ---
        lattice, atom_types, coords = parse_xyz(row["file_path"])

        # 1. Atomic Stream Features
        # One-hot encoding
        one_hot = np.zeros((len(atom_types), NUM_ATOM_TYPES))
        for i, atom in enumerate(atom_types):
            if atom in ATOM_TO_INDEX:
                one_hot[i, ATOM_TO_INDEX[atom]] = 1.0

        # Centered Coordinates
        centered_pos = center_coordinates(coords)

        # Nearest Neighbor Distance (PBC)
        nn_dists = get_pbc_distances(coords, lattice)

        # Concatenate local features: [OneHot(4), Coords(3), NNDist(1)] -> Dim 8
        local_feats = np.hstack([one_hot, centered_pos, nn_dists.reshape(-1, 1)])
        atomic_features_list.append(local_feats.astype(np.float32))

        # --- Global Stream Features ---
        # 1. Geometric (Lattice)
        # Calculate lengths and angles from lattice matrix to be precise, or use CSV.
        # Using CSV columns is standard, but let's calculate volume/density as requested.
        # Lattice lengths
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        # Lattice angles (radians then degrees)
        alpha = np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)) * 180.0 / np.pi
        beta = np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)) * 180.0 / np.pi
        gamma = np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)) * 180.0 / np.pi

        # Volume (scalar triple product)
        volume = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))

        # 2. Structural
        n_atoms = len(atom_types)
        density = n_atoms / volume if volume > 1e-6 else 0.0

        # 3. Chemical (Stoichiometry)
        # Count fractions
        counts = {k: 0 for k in ATOM_TO_INDEX.keys()}
        for at in atom_types:
            if at in counts:
                counts[at] += 1
        frac_al = counts["Al"] / n_atoms
        frac_ga = counts["Ga"] / n_atoms
        frac_in = counts["In"] / n_atoms
        # O is usually dependent, but let's include O fraction or just these 3.
        # Prompt says "Explicit Stoichiometry (percentages of Al, Ga, In)".

        # 4. Physical (Weighted Means)
        phys_means = calculate_physics_features(atom_types)

        # Assemble Global Vector
        # [a, b, c, alpha, beta, gamma, vol, density, n_atoms, f_Al, f_Ga, f_In, mass_avg, rad_avg, en_avg]
        # Dim = 6 + 1 + 1 + 1 + 3 + 3 = 15
        global_vec = np.array(
            [
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
                volume,
                density,
                n_atoms,
                frac_al,
                frac_ga,
                frac_in,
                phys_means[0],
                phys_means[1],
                phys_means[2],
            ]
        )
        global_features_list.append(global_vec)

        # --- Targets ---
        if all(t in row for t in TARGET_COLS):
            # Apply log(1+y) transformation as per strategy
            # Note: Targets are formation_energy and bandgap.
            # Formation energy can be small, bandgap > 0.
            # RMSLE metric implies we should predict log(1+y).
            t_vals = row[TARGET_COLS].values.astype(np.float32)
            targets_list.append(np.log1p(t_vals))
        else:
            targets_list.append(np.array([np.nan, np.nan]))

        ids_list.append(row["id"])

    # Convert to arrays
    # atomic_features_list remains a list because N_atoms varies
    global_features_arr = np.array(global_features_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    ids_arr = np.array(ids_list, dtype=np.int32)

    # 3. Save to Cache
    # We use object array for atomic_features
    atomic_features_obj = np.array(atomic_features_list, dtype=object)

    np.savez_compressed(
        cache_path,
        atomic_features=atomic_features_obj,
        global_features=global_features_arr,
        targets=targets_arr,
        ids=ids_arr,
    )
    print(f"Data processed and saved to {cache_path}")

    return {
        "atomic_features": atomic_features_list,
        "global_features": global_features_arr,
        "targets": targets_arr,
        "ids": ids_arr,
    }


def get_scalers(train_data):
    """
    Computes mean and std for global features from training data for standardization.
    Atomic features (coords, one-hot) are usually handled differently or not scaled
    in the same way (coords are centered, one-hot is 0/1).
    NN dist might need scaling, but often is fine.
    We will scale Global Features and NN Dist (last col of atomic).

    Args:
        train_data (dict): Output of process_dataset for training set.

    Returns:
        scalers (dict): Dictionary containing mean/std for 'global' and 'local_nndist'.
    """
    # Global Scaling
    g_feats = train_data["global_features"]
    g_mean = np.mean(g_feats, axis=0)
    g_std = np.std(g_feats, axis=0)
    g_std[g_std < 1e-6] = 1.0  # Avoid division by zero

    # Local NN Dist Scaling (Index 7 in local feats: 4 onehot + 3 coords + 1 dist)
    # Collect all NN dists
    all_nndists = np.concatenate([f[:, 7] for f in train_data["atomic_features"]])
    l_mean = np.mean(all_nndists)
    l_std = np.std(all_nndists)
    if l_std < 1e-6:
        l_std = 1.0

    return {
        "global_mean": g_mean,
        "global_std": g_std,
        "nndist_mean": l_mean,
        "nndist_std": l_std,
    }


def apply_scalers(data_dict, scalers):
    """
    Applies standardization to data using provided scalers.

    Args:
        data_dict (dict): Data to scale.
        scalers (dict): Scalers from get_scalers.

    Returns:
        scaled_data (dict): Copy of data_dict with scaled features.
    """
    # Deep copy to avoid modifying original
    import copy

    new_data = {
        "ids": data_dict["ids"],
        "targets": data_dict["targets"],
        "global_features": (data_dict["global_features"] - scalers["global_mean"])
        / scalers["global_std"],
        "atomic_features": [],
    }

    g_mean = scalers["global_mean"]
    g_std = scalers["global_std"]
    l_mean = scalers["nndist_mean"]
    l_std = scalers["nndist_std"]

    for feat in data_dict["atomic_features"]:
        # feat is (N, 8). Columns: 0-3 OneHot, 4-6 Coords, 7 NNDist
        # We only scale NNDist. Coords are already centered (spatial invariance).
        # One-hot should stay 0/1.
        new_feat = feat.copy()
        new_feat[:, 7] = (new_feat[:, 7] - l_mean) / l_std
        new_data["atomic_features"].append(new_feat)

    return new_data
