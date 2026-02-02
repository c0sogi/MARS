import os
import numpy as np
import pandas as pd
import torch
from library.config import ATOM_TYPES, PAIR_TYPES, WORKING_DIR, METADATA_DIR, INPUT_DIR
from library.geometry_utils import load_atoms, get_pbc_distances, get_centered_positions


def calculate_cell_volume(a, b, c, alpha, beta, gamma):
    """
    Calculates unit cell volume from lattice lengths and angles (in degrees).
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    term = (
        1
        - np.cos(alpha_rad) ** 2
        - np.cos(beta_rad) ** 2
        - np.cos(gamma_rad) ** 2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )
    return a * b * c * np.sqrt(np.maximum(0, term))


def compute_structural_stats(atoms, distances):
    """
    Computes Mean Pairwise Distances for all unique element pairs.
    Returns a vector of size len(PAIR_TYPES).
    """
    symbols = np.array(atoms.get_chemical_symbols())
    stats_vector = []

    for elem1, elem2 in PAIR_TYPES:
        indices1 = np.where(symbols == elem1)[0]
        indices2 = np.where(symbols == elem2)[0]

        if len(indices1) == 0 or len(indices2) == 0:
            stats_vector.append(0.0)
            continue

        # Extract submatrix of distances
        # We need to be careful not to double count or include self-distance (0) if elem1 == elem2

        if elem1 == elem2:
            # Self-pairs (homo-atomic)
            # Use upper triangle of the submatrix to avoid double counting and diagonal zeros
            # However, get_pbc_distances returns full matrix.
            # For stats, we want average bond length.
            # Extract submatrix
            sub_dists = distances[np.ix_(indices1, indices2)]
            # Get upper triangle indices excluding diagonal
            # Since sub_dists is square and symmetric
            n = len(indices1)
            if n > 1:
                vals = sub_dists[np.triu_indices(n, k=1)]
                mean_dist = np.mean(vals)
            else:
                # Single atom of this type, no pairwise distance
                mean_dist = 0.0
        else:
            # Hetero-atomic
            # Just take the mean of the whole sub-block
            sub_dists = distances[np.ix_(indices1, indices2)]
            mean_dist = np.mean(sub_dists)

        stats_vector.append(mean_dist)

    return np.array(stats_vector, dtype=np.float32)


def extract_atomic_features(atoms, distances):
    """
    Extracts atomic features: One-hot encoding, Centered Coords, NN Distance.
    Returns matrix of shape (N_atoms, 8).
    """
    # 1. One-hot encoding
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    one_hot = np.zeros((n_atoms, len(ATOM_TYPES)), dtype=np.float32)
    for i, s in enumerate(symbols):
        if s in ATOM_TYPES:
            one_hot[i, ATOM_TYPES.index(s)] = 1.0

    # 2. Centered Coordinates
    centered_pos = get_centered_positions(atoms)

    # 3. Nearest Neighbor Distance
    # Mask diagonal with infinity to ignore self-distance
    np.fill_diagonal(distances, np.inf)
    nn_dist = np.min(distances, axis=1).reshape(-1, 1)

    # Concatenate
    # Shape: (N, 4) + (N, 3) + (N, 1) = (N, 8)
    return np.hstack([one_hot, centered_pos, nn_dist])


def extract_global_features(row, structural_stats):
    """
    Extracts global features from metadata row and computed structural stats.
    Returns vector of size 22.
    """
    # Lattice parameters
    a = row["lattice_vector_1_ang"]
    b = row["lattice_vector_2_ang"]
    c = row["lattice_vector_3_ang"]
    alpha = row["lattice_angle_alpha_degree"]
    beta = row["lattice_angle_beta_degree"]
    gamma = row["lattice_angle_gamma_degree"]

    # Derived physics
    vol = calculate_cell_volume(a, b, c, alpha, beta, gamma)
    n_atoms = row["number_of_total_atoms"]
    density = n_atoms / vol if vol > 1e-6 else 0.0

    # Composition
    comp = [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]

    # Combine
    # [3] + [3] + [1] + [1] + [1] + [3] + [10] = 22
    features = np.array(
        [a, b, c, alpha, beta, gamma, vol, density, n_atoms] + comp, dtype=np.float32
    )

    return np.concatenate([features, structural_stats])


def process_split(df, split_name):
    """
    Process a single dataframe split (train/val/test).
    Returns lists of atomic features, global features, and targets (if available).
    """
    atomic_feats_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    print(f"Processing {split_name} split with {len(df)} samples...")

    for idx, row in df.iterrows():
        # Load Geometry
        try:
            atoms = load_atoms(row["file_path"])
        except FileNotFoundError:
            # Fallback if file missing (should not happen based on metadata check)
            print(f"Warning: File not found {row['file_path']}")
            continue

        # Computations
        distances = get_pbc_distances(atoms)

        # Structural Stats (Global)
        struct_stats = compute_structural_stats(atoms, distances)

        # Atomic Features (Local)
        atom_f = extract_atomic_features(atoms, distances)

        # Global Features
        glob_f = extract_global_features(row, struct_stats)

        # Targets
        if "formation_energy_ev_natom" in row and not pd.isna(
            row["formation_energy_ev_natom"]
        ):
            target = np.array(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]],
                dtype=np.float32,
            )
            targets_list.append(target)
        else:
            # Placeholder for test set
            targets_list.append(np.array([0.0, 0.0], dtype=np.float32))

        atomic_feats_list.append(atom_f)
        global_feats_list.append(glob_f)
        ids_list.append(row["id"])

    return (
        atomic_feats_list,
        np.array(global_feats_list),
        np.array(targets_list),
        np.array(ids_list),
    )


class StandardScaler:
    """Simple standard scaler for numpy arrays."""

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Prevent division by zero
        self.scale[self.scale < 1e-8] = 1.0

    def transform(self, X):
        return (X - self.mean) / self.scale


def prepare_data(load_cached_data=True):
    """
    Main function to load, process, scale, and cache data.
    """
    cache_files = {
        "train": os.path.join(WORKING_DIR, "train_data.npz"),
        "val": os.path.join(WORKING_DIR, "val_data.npz"),
        "test": os.path.join(WORKING_DIR, "test_data.npz"),
        "scalers": os.path.join(WORKING_DIR, "scalers.npz"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        data = {}
        for split in ["train", "val", "test"]:
            loaded = np.load(cache_files[split], allow_pickle=True)
            # atomic features are stored as object array of arrays
            data[split] = {
                "atomic": loaded["atomic"],
                "global": loaded["global"],
                "targets": loaded["targets"],
                "ids": loaded["ids"],
            }
        return data

    print("Computing features from scratch...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Process splits
    train_atomic, train_global, train_targets, train_ids = process_split(
        train_df, "Train"
    )
    val_atomic, val_global, val_targets, val_ids = process_split(val_df, "Validation")
    test_atomic, test_global, test_targets, test_ids = process_split(test_df, "Test")

    # --- Scaling ---
    print("Fitting scalers on training data...")

    # 1. Scale Atomic Features (Indices 4-7: x, y, z, nn_dist)
    # Flatten train atomic features to fit scaler
    all_train_atomic = np.vstack(train_atomic)
    atomic_scaler = StandardScaler()
    # Only scale continuous columns (4, 5, 6, 7)
    atomic_scaler.fit(all_train_atomic[:, 4:])

    # 2. Scale Global Features (All 22 dims)
    global_scaler = StandardScaler()
    global_scaler.fit(train_global)

    def apply_scaling(atomic_list, global_arr):
        scaled_atomic = []
        for feat in atomic_list:
            feat_copy = feat.copy()
            feat_copy[:, 4:] = atomic_scaler.transform(feat[:, 4:])
            scaled_atomic.append(feat_copy)

        scaled_global = global_scaler.transform(global_arr)
        return np.array(scaled_atomic, dtype=object), scaled_global

    print("Applying scaling...")
    train_atomic_s, train_global_s = apply_scaling(train_atomic, train_global)
    val_atomic_s, val_global_s = apply_scaling(val_atomic, val_global)
    test_atomic_s, test_global_s = apply_scaling(test_atomic, test_global)

    # --- Target Transformation ---
    # Log1p transformation for targets
    print("Transforming targets (Log1p)...")
    train_targets_s = np.log1p(train_targets)
    val_targets_s = np.log1p(val_targets)
    # Test targets are placeholders, no need to transform

    # --- Caching ---
    print("Caching processed data...")
    np.savez(
        cache_files["train"],
        atomic=train_atomic_s,
        global_=train_global_s,
        targets=train_targets_s,
        ids=train_ids,
    )
    np.savez(
        cache_files["val"],
        atomic=val_atomic_s,
        global_=val_global_s,
        targets=val_targets_s,
        ids=val_ids,
    )
    np.savez(
        cache_files["test"],
        atomic=test_atomic_s,
        global_=test_global_s,
        targets=test_targets,
        ids=test_ids,
    )

    # Save scaler params for potential inverse transform or inference usage later
    np.savez(
        cache_files["scalers"],
        atomic_mean=atomic_scaler.mean,
        atomic_scale=atomic_scaler.scale,
        global_mean=global_scaler.mean,
        global_scale=global_scaler.scale,
    )

    data = {
        "train": {
            "atomic": train_atomic_s,
            "global": train_global_s,
            "targets": train_targets_s,
            "ids": train_ids,
        },
        "val": {
            "atomic": val_atomic_s,
            "global": val_global_s,
            "targets": val_targets_s,
            "ids": val_ids,
        },
        "test": {
            "atomic": test_atomic_s,
            "global": test_global_s,
            "targets": test_targets,
            "ids": test_ids,
        },
    }

    return data
