import os
import numpy as np
import pandas as pd
from library.utils import (
    ATOMIC_PROPS,
    calculate_cell_volume,
    calculate_angular_distortion,
    get_pbc_distances,
)

# Constants
ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = 4
CACHE_DIR = "./working/idea_47/"


def parse_xyz(file_path):
    """Parses a geometry.xyz file to extract lattice and atomic information."""
    lattice = []
    atoms = []
    coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            coords.append([float(x) for x in parts[1:4]])
            atoms.append(parts[4])

    return np.array(lattice), atoms, np.array(coords)


def get_one_hot(atom_symbol):
    """Returns a one-hot vector for a given atom symbol."""
    vec = np.zeros(NUM_ATOM_TYPES, dtype=np.float32)
    if atom_symbol in ATOM_MAP:
        vec[ATOM_MAP[atom_symbol]] = 1.0
    return vec


def compute_weighted_context(neighbor_indices, neighbor_dists, atom_symbols):
    """Computes inverse-distance weighted chemical composition vector."""
    # neighbor_indices: (K,)
    # neighbor_dists: (K,)

    context_vec = np.zeros(NUM_ATOM_TYPES, dtype=np.float32)
    total_weight = 0.0

    for idx, dist in zip(neighbor_indices, neighbor_dists):
        if dist < 1e-6:
            continue  # Avoid division by zero
        weight = 1.0 / dist
        symbol = atom_symbols[idx]
        context_vec += weight * get_one_hot(symbol)
        total_weight += weight

    if total_weight > 0:
        context_vec /= total_weight

    return context_vec


def get_physical_stats(atom_symbols):
    """Computes mean and std of physical properties for the atoms in the cell."""
    masses = []
    radii = []
    electronegs = []

    for sym in atom_symbols:
        props = ATOMIC_PROPS.get(sym, {})
        if props:
            masses.append(props.get("mass", 0.0))
            radii.append(props.get("radius", 0.0))
            electronegs.append(props.get("electronegativity", 0.0))

    if not masses:
        return np.zeros(6, dtype=np.float32)

    stats = [
        np.mean(masses),
        np.std(masses),
        np.mean(radii),
        np.std(radii),
        np.mean(electronegs),
        np.std(electronegs),
    ]
    return np.array(stats, dtype=np.float32)


def extract_atomic_features(coords, lattice, atom_symbols):
    """
    Generates 21-dim atomic features for all atoms in a crystal.
    """
    n_atoms = len(atom_symbols)

    # Centering
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # Neighbor search (K=24 to cover all needs)
    # K=1 for NN, K=12 for packing, K=6 and K=24 for context
    k_max = min(24, n_atoms - 1)  # Handle small systems if any
    if k_max < 1:
        # Fallback for single atom case (unlikely in this dataset)
        return np.zeros((n_atoms, 21), dtype=np.float32)

    dists, indices = get_pbc_distances(coords, lattice, k_max)

    features = []

    for i in range(n_atoms):
        # 1. Atomic Identity (4)
        feat_identity = get_one_hot(atom_symbols[i])

        # 2. Nearest Neighbor Identity (4)
        nn_idx = indices[i, 0]
        feat_nn_identity = get_one_hot(atom_symbols[nn_idx])

        # 3. Spatial Context (3)
        feat_spatial = centered_coords[i]

        # 4. Nearest Neighbor Distance (1)
        d_min = dists[i, 0]

        # 5. Local Packing Ratio (1)
        # Use up to K=12 for mean distance
        k_pack = min(12, k_max)
        d_mean_12 = np.mean(dists[i, :k_pack])
        packing_ratio = d_min / (d_mean_12 + 1e-8)

        # 6. Multi-Scale Chemical Contexts (8)
        # Short range K=6
        k_short = min(6, k_max)
        ctx_short = compute_weighted_context(
            indices[i, :k_short], dists[i, :k_short], atom_symbols
        )

        # Medium range K=24
        k_med = min(24, k_max)
        ctx_med = compute_weighted_context(
            indices[i, :k_med], dists[i, :k_med], atom_symbols
        )

        # Concatenate
        atom_feat = np.concatenate(
            [
                feat_identity,  # 0-3
                feat_nn_identity,  # 4-7
                feat_spatial,  # 8-10
                [d_min],  # 11
                [packing_ratio],  # 12
                ctx_short,  # 13-16
                ctx_med,  # 17-20
            ]
        )
        features.append(atom_feat)

    return np.array(features, dtype=np.float32)


def extract_global_features(row, atom_symbols):
    """
    Generates global descriptor vector.
    """
    # 1. Geometric
    a, b, c = (
        row["lattice_vector_1_ang"],
        row["lattice_vector_2_ang"],
        row["lattice_vector_3_ang"],
    )
    alpha, beta, gamma = (
        row["lattice_angle_alpha_degree"],
        row["lattice_angle_beta_degree"],
        row["lattice_angle_gamma_degree"],
    )

    vol = calculate_cell_volume(a, b, c, alpha, beta, gamma)

    # Aspect ratios
    ar1 = a / (b + 1e-8)
    ar2 = b / (c + 1e-8)
    ar3 = c / (a + 1e-8)

    # Angular distortion
    ang_dist = calculate_angular_distortion(alpha, beta, gamma)

    # 2. Structural
    n_atoms = row["number_of_total_atoms"]
    density = n_atoms / (vol + 1e-8)

    # 3. Chemical (Stoichiometry from CSV)
    stoich = np.array(
        [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]],
        dtype=np.float32,
    )

    # 4. Physical Statistics (Mean & Std of Mass, Radius, Electronegativity)
    phys_stats = get_physical_stats(atom_symbols)

    # Concatenate
    # Lattice (6) + Vol (1) + AR (3) + Dist (1) + Density (1) + N_atoms (1) + Stoich (3) + Phys (6) = 22
    global_feat = np.concatenate(
        [
            [a, b, c, alpha, beta, gamma],
            [vol],
            [ar1, ar2, ar3],
            [ang_dist],
            [density],
            [n_atoms],
            stoich,
            phys_stats,
        ]
    )

    return global_feat


def process_dataset(metadata_path, input_dir):
    """
    Processes a dataset defined by a metadata CSV file.
    Returns lists/arrays suitable for sparse batching.
    """
    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    all_batch_indices = []
    all_global_features = []
    all_targets = []
    all_ids = []

    # Check if targets exist
    has_targets = "formation_energy_ev_natom" in df.columns

    for idx, row in df.iterrows():
        crystal_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}")
            continue

        # Parse Geometry
        lattice, atom_symbols, coords = parse_xyz(full_path)

        # Atomic Features
        atomic_feats = extract_atomic_features(coords, lattice, atom_symbols)
        n_atoms = len(atomic_feats)

        # Batch Indices (maps atoms to crystal index in the batch)
        # Here we map to the index in the current dataset (0 to N-1)
        batch_idx = np.full(n_atoms, idx, dtype=np.int32)

        # Global Features
        global_feats = extract_global_features(row, atom_symbols)

        # Targets
        if has_targets:
            # Log transform targets: log(1 + y)
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            targets = np.array([t1, t2], dtype=np.float32)
        else:
            targets = np.array([0.0, 0.0], dtype=np.float32)  # Placeholder

        all_atomic_features.append(atomic_feats)
        all_batch_indices.append(batch_idx)
        all_global_features.append(global_feats)
        all_targets.append(targets)
        all_ids.append(crystal_id)

    # Concatenate
    # Atomic features: (Total_Atoms, 21)
    X_atomic = np.vstack(all_atomic_features)
    # Batch indices: (Total_Atoms,) - Note: these are 0-based indices relative to the dataset
    batch_indices = np.concatenate(all_batch_indices)
    # Global features: (N_crystals, 22)
    X_global = np.vstack(all_global_features)
    # Targets: (N_crystals, 2)
    y = np.vstack(all_targets)
    # IDs
    ids = np.array(all_ids, dtype=np.int32)

    return X_atomic, batch_indices, X_global, y, ids


def load_and_process_data(
    split_name, input_dir="./input", metadata_dir="./metadata", load_cached_data=True
):
    """
    Main entry point for data loading.
    split_name: 'train', 'val', or 'test'
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{split_name}_data.npz")
    scaler_path = os.path.join(CACHE_DIR, "scalers.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache...")
        data = np.load(cache_path)
        return {
            "atomic_features": data["atomic_features"],
            "batch_indices": data["batch_indices"],
            "global_features": data["global_features"],
            "targets": data["targets"],
            "ids": data["ids"],
        }

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch...")
    metadata_path = os.path.join(metadata_dir, f"{split_name}.csv")
    X_atomic, batch_indices, X_global, y, ids = process_dataset(
        metadata_path, input_dir
    )

    # 3. Scaling
    # We need to scale continuous features.
    # Atomic continuous indices: 8, 9, 10 (coords), 11 (d_min), 12 (packing), 13-20 (context - weighted comp)
    # Note: Context vectors are 0-1 bounded, but standard scaling usually helps.
    # One-hot indices (0-7) should NOT be scaled.
    atomic_continuous_cols = list(range(8, 21))

    # Global continuous indices: All 22 are continuous-ish. Stoichiometry (13-15) is 0-1.
    # We will scale all global features.

    if split_name == "train":
        # Compute scalers
        atomic_mean = np.mean(X_atomic[:, atomic_continuous_cols], axis=0)
        atomic_std = np.std(X_atomic[:, atomic_continuous_cols], axis=0)
        # Avoid div by zero
        atomic_std[atomic_std < 1e-8] = 1.0

        global_mean = np.mean(X_global, axis=0)
        global_std = np.std(X_global, axis=0)
        global_std[global_std < 1e-8] = 1.0

        # Save scalers
        np.savez(
            scaler_path,
            atomic_mean=atomic_mean,
            atomic_std=atomic_std,
            global_mean=global_mean,
            global_std=global_std,
        )
    else:
        # Load scalers
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scalers not found! Process 'train' split first.")
        scalers = np.load(scaler_path)
        atomic_mean = scalers["atomic_mean"]
        atomic_std = scalers["atomic_std"]
        global_mean = scalers["global_mean"]
        global_std = scalers["global_std"]

    # Apply Scaling
    X_atomic[:, atomic_continuous_cols] = (
        X_atomic[:, atomic_continuous_cols] - atomic_mean
    ) / atomic_std
    X_global = (X_global - global_mean) / global_std

    # 4. Save to cache
    np.savez(
        cache_path,
        atomic_features=X_atomic,
        batch_indices=batch_indices,
        global_features=X_global,
        targets=y,
        ids=ids,
    )

    return {
        "atomic_features": X_atomic,
        "batch_indices": batch_indices,
        "global_features": X_global,
        "targets": y,
        "ids": ids,
    }
