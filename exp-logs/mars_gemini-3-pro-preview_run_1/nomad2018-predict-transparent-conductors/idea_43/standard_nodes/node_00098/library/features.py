import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import (
    get_atomic_one_hot,
    get_pbc_neighbors,
    center_coordinates,
    compute_pbc_distance_matrix,
)


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic symbols, and coordinates.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    lattice_vectors = []
    atom_symbols = []
    atom_coords = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            atom_coords.append([float(x) for x in parts[1:4]])
            atom_symbols.append(parts[4])

    # Ensure coords has shape (N, 3) even if empty to prevent broadcasting errors
    coords = np.array(atom_coords)
    if coords.size == 0:
        coords = coords.reshape(0, 3)

    return np.array(lattice_vectors), atom_symbols, coords


def extract_atomic_features(atom_symbols, atom_coords, lattice):
    """
    Extracts distortion-aware point features for each atom.
    """
    n_atoms = len(atom_symbols)

    # 1. Centered Coordinates
    centered_coords = center_coordinates(atom_coords)

    # 2. Neighbor Search
    # We need neighbors for K=1 (d_min), K=12 (packing), K=6 (context), K=24 (context)
    max_k = max(1, 12, Config.K_NEIGHBORS_SHORT, Config.K_NEIGHBORS_MEDIUM)

    # Get neighbors (indices and distances)
    # Note: get_pbc_neighbors returns sorted neighbors
    neighbor_indices, neighbor_dists = get_pbc_neighbors(atom_coords, lattice, max_k)

    features = []

    for i in range(n_atoms):
        # A. Atomic Identity (One-hot) - 4 dims
        identity = get_atomic_one_hot(atom_symbols[i])

        # B. Spatial Context - 3 dims
        spatial = centered_coords[i]

        # C. Nearest Neighbor Distance (d_min) - 1 dim
        # neighbor_dists[i, 0] is the closest neighbor distance
        d_min = neighbor_dists[i, 0]

        # D. Local Packing Ratio - 1 dim
        # Mean distance to K=12 neighbors
        k_packing = 12
        d_mean_12 = np.mean(neighbor_dists[i, :k_packing])
        # Avoid division by zero if something goes wrong, though unlikely in crystals
        packing_ratio = d_min / (d_mean_12 + 1e-8)

        # E. Multi-Scale Chemical Contexts
        # Helper to compute weighted context
        def get_context(k_neighbors):
            indices = neighbor_indices[i, :k_neighbors]
            dists = neighbor_dists[i, :k_neighbors]

            # Inverse distance weighting (add epsilon to avoid div by zero)
            weights = 1.0 / (dists + 1e-6)
            weights /= np.sum(weights)  # Normalize weights

            # Weighted sum of neighbor one-hots
            context_vec = np.zeros(4, dtype=np.float32)
            for idx, w in zip(indices, weights):
                # Map neighbor index back to original atom index (modulo N is handled by get_pbc_neighbors logic usually returning valid indices,
                # but get_pbc_neighbors implementation in utils returns indices in 0..N-1 range directly)
                neighbor_sym = atom_symbols[idx % n_atoms]
                context_vec += w * get_atomic_one_hot(neighbor_sym)
            return context_vec

        context_short = get_context(Config.K_NEIGHBORS_SHORT)  # 4 dims
        context_medium = get_context(Config.K_NEIGHBORS_MEDIUM)  # 4 dims

        # Concatenate all features
        # [Identity(4), Spatial(3), d_min(1), Ratio(1), ContextS(4), ContextM(4)] = 17 dims
        atom_feat = np.concatenate(
            [identity, spatial, [d_min], [packing_ratio], context_short, context_medium]
        )
        features.append(atom_feat)

    if not features:
        # Handle empty case: return empty array with correct feature dimension (17)
        return np.zeros((0, 17), dtype=np.float32)

    return np.array(features, dtype=np.float32)


def extract_global_features(atom_symbols, lattice):
    """
    Extracts anisotropic physics context features for the crystal.
    """
    # 1. Lattice Parameters
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])

    # Angles
    def angle(v1, v2):
        return np.degrees(
            np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        )

    alpha = angle(lattice[1], lattice[2])
    beta = angle(lattice[0], lattice[2])
    gamma = angle(lattice[0], lattice[1])

    # Volume
    volume = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))

    # 2. Atomic Density
    n_atoms = len(atom_symbols)
    density = n_atoms / volume

    # 3. Lattice Aspect Ratios (Anisotropy)
    ratio_ab = a / b
    ratio_bc = b / c
    ratio_ca = c / a

    # 4. Stoichiometry and Weighted Physics
    # Count atoms
    counts = {"Al": 0, "Ga": 0, "In": 0, "O": 0}
    for s in atom_symbols:
        if s in counts:
            counts[s] += 1

    # Fractions
    if n_atoms > 0:
        fracs = {k: v / n_atoms for k, v in counts.items()}
    else:
        fracs = {k: 0.0 for k in counts.keys()}

    # Weighted Properties
    # Config.ATOMIC_PROPS: "Symbol": [Mass, Radius, Electronegativity]
    avg_mass = 0.0
    avg_radius = 0.0
    avg_eneg = 0.0

    for sym, frac in fracs.items():
        props = Config.ATOMIC_PROPS.get(sym, [0, 0, 0])
        avg_mass += frac * props[0]
        avg_radius += frac * props[1]
        avg_eneg += frac * props[2]

    # Construct feature vector
    # [a, b, c, alpha, beta, gamma, vol, dens, fracAl, fracGa, fracIn, N, ab, bc, ca, mass, rad, eneg]
    # Note: fracO is redundant if others sum to 1 (mostly), but we include cations explicitly.
    # Total Atoms is included.

    features = np.array(
        [
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            volume,
            density,
            fracs["Al"],
            fracs["Ga"],
            fracs["In"],
            float(n_atoms),
            ratio_ab,
            ratio_bc,
            ratio_ca,
            avg_mass,
            avg_radius,
            avg_eneg,
        ],
        dtype=np.float32,
    )

    return features


def process_dataset(metadata_path, cache_path, load_cached=True):
    """
    Processes a dataset (train/val/test) extracting atomic and global features.
    Handles caching.
    """
    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return np.load(cache_path)

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    batch_indices = []
    all_global_features = []
    all_targets = []
    ids = []

    # Check if targets exist
    has_targets = "formation_energy_ev_natom" in df.columns

    for idx, row in df.iterrows():
        # Parse geometry
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        lattice, symbols, coords = parse_xyz(full_path)

        # Extract features
        atomic_feats = extract_atomic_features(symbols, coords, lattice)
        global_feats = extract_global_features(symbols, lattice)

        # Accumulate
        n_atoms = len(symbols)
        all_atomic_features.append(atomic_feats)
        batch_indices.append(
            np.full(n_atoms, idx, dtype=int)
        )  # Map atoms to crystal index (0 to N_samples-1)
        all_global_features.append(global_feats)
        ids.append(row["id"])

        if has_targets:
            # Log transform targets: log(1 + y)
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            all_targets.append([t1, t2])

    # Concatenate
    atomic_features_flat = np.vstack(all_atomic_features)
    batch_indices_flat = np.concatenate(batch_indices)
    global_features_stacked = np.vstack(all_global_features)

    if has_targets:
        targets_stacked = np.vstack(all_targets)
    else:
        targets_stacked = np.zeros((len(df), 2), dtype=np.float32)  # Placeholder

    ids_array = np.array(ids, dtype=int)

    # Save to cache
    np.savez(
        cache_path,
        atomic_features=atomic_features_flat,
        batch_indices=batch_indices_flat,
        global_features=global_features_stacked,
        targets=targets_stacked,
        ids=ids_array,
    )

    print(f"Saved processed data to {cache_path}")

    return np.load(cache_path)


def load_and_preprocess_data(load_cached=True):
    """
    Main entry point to load all datasets and apply scaling.
    """
    # 1. Load/Process Raw Data
    train_data = process_dataset(
        Config.TRAIN_METADATA, Config.TRAIN_DATA_CACHE, load_cached
    )
    val_data = process_dataset(Config.VAL_METADATA, Config.VAL_DATA_CACHE, load_cached)
    test_data = process_dataset(
        Config.TEST_METADATA, Config.TEST_DATA_CACHE, load_cached
    )

    # 2. Scaling
    # We need to fit scalers on Train data and apply to all.

    # A. Atomic Features Scaling
    # Indices 0-3 are One-Hot (Identity) -> DO NOT SCALE
    # Indices 4-16 are Continuous -> SCALE

    train_atomic = train_data["atomic_features"]

    # Split atomic features
    train_atomic_cat = train_atomic[:, :4]
    train_atomic_cont = train_atomic[:, 4:]

    atomic_scaler = StandardScaler()
    atomic_scaler.fit(train_atomic_cont)

    # B. Global Features Scaling
    # All global features are continuous -> SCALE ALL
    train_global = train_data["global_features"]
    global_scaler = StandardScaler()
    global_scaler.fit(train_global)

    # Save scalers
    np.savez(
        Config.SCALERS_CACHE,
        atomic_mean=atomic_scaler.mean_,
        atomic_scale=atomic_scaler.scale_,
        global_mean=global_scaler.mean_,
        global_scale=global_scaler.scale_,
    )

    # Helper to apply scaling
    def apply_scaling(data_obj):
        # Atomic
        atomic = data_obj["atomic_features"]
        atomic_cat = atomic[:, :4]
        atomic_cont = atomic[:, 4:]
        atomic_cont_scaled = atomic_scaler.transform(atomic_cont)
        atomic_scaled = np.hstack([atomic_cat, atomic_cont_scaled])

        # Global
        glob = data_obj["global_features"]
        glob_scaled = global_scaler.transform(glob)

        return {
            "atomic_features": atomic_scaled.astype(np.float32),
            "batch_indices": data_obj["batch_indices"],
            "global_features": glob_scaled.astype(np.float32),
            "targets": data_obj["targets"].astype(np.float32),
            "ids": data_obj["ids"],
        }

    train_scaled = apply_scaling(train_data)
    val_scaled = apply_scaling(val_data)
    test_scaled = apply_scaling(test_data)

    return train_scaled, val_scaled, test_scaled
