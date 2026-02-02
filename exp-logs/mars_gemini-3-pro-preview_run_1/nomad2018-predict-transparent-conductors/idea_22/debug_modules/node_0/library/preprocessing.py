import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import StandardScaler


def parse_xyz(file_path):
    """
    Parses the custom geometry.xyz file format.

    Args:
        file_path: Path to the .xyz file.

    Returns:
        lattice: np.ndarray of shape (3, 3) containing lattice vectors.
        species: List[str] of atomic species.
        coords: np.ndarray of shape (N, 3) containing atomic coordinates.
    """
    lattice_vectors = []
    atom_species = []
    atom_coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "lattice_vector":
            vec = [float(x) for x in parts[1:4]]
            lattice_vectors.append(vec)
        elif parts[0] == "atom":
            # Format: atom x y z species
            pos = [float(x) for x in parts[1:4]]
            spec = parts[4]
            atom_coords.append(pos)
            atom_species.append(spec)

    return np.array(lattice_vectors), atom_species, np.array(atom_coords)


def compute_pbc_neighbor_stats(coords, lattice, k=12):
    """
    Computes distance to nearest neighbor (d_min) and mean distance to
    K nearest neighbors (d_mean_k) respecting periodic boundary conditions.

    Args:
        coords: (N, 3) array of atomic coordinates.
        lattice: (3, 3) array of lattice vectors.
        k: Number of neighbors to consider for mean distance.

    Returns:
        d_min: (N,) array of minimum distances.
        d_mean_k: (N,) array of mean distances to k neighbors.
    """
    n_atoms = len(coords)

    # Generate neighbor images (3x3x3 supercell)
    # Offsets: -1, 0, 1 for each lattice vector
    ranges = [-1, 0, 1]
    offsets = np.array([[i, j, l] for i in ranges for j in ranges for l in ranges])
    # Shape: (27, 3)

    # Translation vectors: (27, 3)
    translations = offsets @ lattice

    # Create supercell coordinates: (27 * N, 3)
    # Broadcast addition: (N, 1, 3) + (1, 27, 3) -> (N, 27, 3) -> reshape
    supercell_coords = (coords[:, None, :] + translations[None, :, :]).reshape(-1, 3)

    # Compute distances from original atoms to all atoms in supercell
    # Dist matrix: (N, 27*N)
    # Using broadcasting: (N, 1, 3) - (1, 27*N, 3)
    delta = coords[:, None, :] - supercell_coords[None, :, :]
    dists = np.sqrt(np.sum(delta**2, axis=2))

    # For each atom, sort distances
    # The first distance is always 0 (self-distance to image 0,0,0), so we ignore it
    sorted_dists = np.sort(dists, axis=1)

    # Neighbors start from index 1
    # We need k neighbors. If system is small (N < k), we still use supercell which has 27N atoms.
    # 27N is typically >> k.

    # d_min is the distance to the closest non-self atom (index 1)
    d_min = sorted_dists[:, 1]

    # d_mean_k is average of distances to neighbors 1 to k
    # Ensure we don't go out of bounds if for some reason 27N <= k (unlikely)
    actual_k = min(k, sorted_dists.shape[1] - 1)
    d_mean_k = np.mean(sorted_dists[:, 1 : 1 + actual_k], axis=1)

    return d_min, d_mean_k


def get_global_features(row, lattice, n_atoms):
    """
    Extracts global thermodynamic features from metadata row and lattice.

    Args:
        row: Pandas Series containing metadata.
        lattice: (3, 3) lattice matrix.
        n_atoms: Total number of atoms.

    Returns:
        np.ndarray of shape (12,) containing global features.
    """
    # 1. Lattice lengths
    lengths = np.linalg.norm(lattice, axis=1)

    # 2. Lattice angles (provided in metadata, but can be computed)
    # Using metadata values for consistency
    angles = np.array(
        [
            row["lattice_angle_alpha_degree"],
            row["lattice_angle_beta_degree"],
            row["lattice_angle_gamma_degree"],
        ]
    )

    # 3. Volume
    # Scalar triple product: det(lattice)
    volume = np.abs(np.linalg.det(lattice))

    # 4. Atomic Density
    density = n_atoms / volume if volume > 1e-6 else 0.0

    # 5. Stoichiometry
    stoich = np.array(
        [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
    )

    # 6. Total Atoms
    total_atoms = float(n_atoms)

    # Combine
    return np.concatenate([lengths, angles, [volume, density], stoich, [total_atoms]])


def extract_features(df, root_dir):
    """
    Main loop to extract features for a dataset.

    Args:
        df: DataFrame containing metadata.
        root_dir: Root directory for file paths.

    Returns:
        atomic_feats_list: List of np.ndarrays (N_i, 9)
        global_feats_array: np.ndarray (Batch, 12)
        targets_array: np.ndarray (Batch, 2) or None
        ids: np.ndarray (Batch,)
    """
    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    atom_types_map = {atype: i for i, atype in enumerate(Config.ATOM_TYPES)}

    for _, row in df.iterrows():
        # Load Geometry
        full_path = os.path.join(root_dir, row["file_path"])
        lattice, species, coords = parse_xyz(full_path)
        n_atoms = len(species)

        # --- Atomic Stream Features ---
        # 1. One-hot encoding
        one_hot = np.zeros((n_atoms, len(Config.ATOM_TYPES)))
        for i, s in enumerate(species):
            if s in atom_types_map:
                one_hot[i, atom_types_map[s]] = 1.0

        # 2. Centered Coordinates
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # 3. Neighbor Stats
        d_min, d_mean_k = compute_pbc_neighbor_stats(
            coords, lattice, k=Config.K_NEIGHBORS
        )

        # Concatenate Atomic Features: (N, 4) + (N, 3) + (N, 1) + (N, 1) = (N, 9)
        # Reshape scalars to (N, 1)
        atom_f = np.concatenate(
            [one_hot, centered_coords, d_min[:, None], d_mean_k[:, None]], axis=1
        )

        atomic_feats_list.append(atom_f.astype(np.float32))

        # --- Global Stream Features ---
        glob_f = get_global_features(row, lattice, n_atoms)
        global_feats_list.append(glob_f.astype(np.float32))

        # --- Targets ---
        if all(c in row for c in Config.TARGET_COLS):
            t = row[Config.TARGET_COLS].values.astype(np.float32)
            targets_list.append(t)
        else:
            # For test set, placeholders
            targets_list.append(np.array([0.0, 0.0], dtype=np.float32))

        ids_list.append(row["id"])

    return (
        atomic_feats_list,
        np.array(global_feats_list),
        np.array(targets_list),
        np.array(ids_list),
    )


def prepare_data(load_cached=True):
    """
    Orchestrates data loading, feature extraction, scaling, and caching.

    Args:
        load_cached: If True, attempts to load pre-processed .npz files.

    Returns:
        train_data, val_data, test_data
        Each is a dictionary containing:
          'atomic': List of arrays
          'global': Array
          'targets': Array
          'ids': Array
    """
    cache_dir = Config.WORKING_DIR
    train_cache = os.path.join(cache_dir, "train_data.npz")
    val_cache = os.path.join(cache_dir, "val_data.npz")
    test_cache = os.path.join(cache_dir, "test_data.npz")
    scaler_cache = os.path.join(cache_dir, "scalers.npz")

    # Try loading cached data
    if (
        load_cached
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(scaler_cache)
    ):
        print("Loading cached data...")
        try:
            # Helper to load npz to dict
            def load_npz(path):
                data = np.load(path, allow_pickle=True)
                return {
                    "atomic": list(data["atomic"]),  # Convert back to list of arrays
                    "global": data["global"],
                    "targets": data["targets"],
                    "ids": data["ids"],
                }

            return load_npz(train_cache), load_npz(val_cache), load_npz(test_cache)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print("Computing features from scratch...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Extract raw features
    print("Extracting Train...")
    train_atomic, train_global, train_targets, train_ids = extract_features(
        train_df, Config.INPUT_DIR
    )
    print("Extracting Val...")
    val_atomic, val_global, val_targets, val_ids = extract_features(
        val_df, Config.INPUT_DIR
    )
    print("Extracting Test...")
    test_atomic, test_global, test_targets, test_ids = extract_features(
        test_df, Config.INPUT_DIR
    )

    # Initialize Scalers
    # Atomic features to scale: indices 4 to 8 (coords: 4,5,6; d_min: 7; d_mean: 8)
    # One-hot (0-3) is not scaled.
    atomic_scaler = StandardScaler()
    global_scaler = StandardScaler()

    # Collect all atomic features for fitting scaler (flatten list)
    all_train_atomic = np.concatenate(train_atomic, axis=0)

    # Fit scalers on training data only
    # Scale only continuous atomic features (cols 4:)
    atomic_scaler.fit(all_train_atomic[:, 4:])
    global_scaler.fit(train_global)

    # Transform function
    def transform_atomic(feat_list, scaler):
        scaled_list = []
        for arr in feat_list:
            # Copy to avoid modifying original if needed, though we overwrite
            new_arr = arr.copy()
            new_arr[:, 4:] = scaler.transform(arr[:, 4:])
            scaled_list.append(new_arr)
        return scaled_list

    def transform_global(feat_array, scaler):
        return scaler.transform(feat_array)

    # Apply scaling
    train_atomic = transform_atomic(train_atomic, atomic_scaler)
    val_atomic = transform_atomic(val_atomic, atomic_scaler)
    test_atomic = transform_atomic(test_atomic, atomic_scaler)

    train_global = transform_global(train_global, global_scaler)
    val_global = transform_global(val_global, global_scaler)
    test_global = transform_global(test_global, global_scaler)

    # Target Transformation (Log1p)
    if Config.LOG_TARGETS:
        train_targets = np.log1p(train_targets)
        val_targets = np.log1p(val_targets)
        # Test targets are dummy, no need to transform, but for consistency we can leave them

    # Save scalers
    # We save mean/scale manually to npz for simplicity alongside data
    np.savez(
        scaler_cache,
        atomic_mean=atomic_scaler.mean,
        atomic_scale=atomic_scaler.scale,
        global_mean=global_scaler.mean,
        global_scale=global_scaler.scale,
    )

    # Save Data
    def save_npz(path, atomic, glob, targ, ids):
        # atomic is list of arrays, np.savez handles object arrays
        # We cast atomic list to object array
        atomic_obj = np.array(atomic + [None], dtype=object)[:-1]
        np.savez(path, atomic=atomic_obj, global_data=glob, targets=targ, ids=ids)

    # Note: Using keyword 'global_data' because 'global' is a keyword
    # But for consistency with load function, I will map keys.
    # np.savez saves with keyword arguments as keys.

    np.savez(
        train_cache,
        atomic=np.array(train_atomic, dtype=object),
        **{"global": train_global, "targets": train_targets, "ids": train_ids},
    )
    np.savez(
        val_cache,
        atomic=np.array(val_atomic, dtype=object),
        **{"global": val_global, "targets": val_targets, "ids": val_ids},
    )
    np.savez(
        test_cache,
        atomic=np.array(test_atomic, dtype=object),
        **{"global": test_global, "targets": test_targets, "ids": test_ids},
    )

    print("Data processing complete and cached.")

    return (
        {
            "atomic": train_atomic,
            "global": train_global,
            "targets": train_targets,
            "ids": train_ids,
        },
        {
            "atomic": val_atomic,
            "global": val_global,
            "targets": val_targets,
            "ids": val_ids,
        },
        {
            "atomic": test_atomic,
            "global": test_global,
            "targets": test_targets,
            "ids": test_ids,
        },
    )
