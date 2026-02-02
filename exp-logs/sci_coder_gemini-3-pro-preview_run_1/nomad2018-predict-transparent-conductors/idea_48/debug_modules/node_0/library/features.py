import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from library.config import (
    ATOMIC_MASSES,
    COVALENT_RADII,
    ELECTRONEGATIVITY,
    ATOMIC_LABELS,
    ATOM_TO_INDEX,
    INPUT_DIR,
    WORKING_DIR,
    NEIGHBOR_K_SHORT,
    NEIGHBOR_K_LONG,
    ATOMIC_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
)


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file.
    Returns:
        lattice_vectors (3x3 np.array)
        atom_types (list of str)
        atom_coords (Nx3 np.array)
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    with open(full_path, "r") as f:
        lines = f.readlines()

    lattice_vectors = []
    atom_types = []
    atom_coords = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            atom_coords.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(atom_coords)


def compute_pbc_neighbors(lattice_vectors, atom_coords, atom_types, k_max):
    """
    Finds nearest neighbors under PBC.
    Returns:
        neighbors_dist: (N, k_max)
        neighbors_types: (N, k_max) list of types
    """
    n_atoms = len(atom_coords)

    # Create supercell (3x3x3)
    # Range -1, 0, 1
    translations = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                translations.append(
                    i * lattice_vectors[0]
                    + j * lattice_vectors[1]
                    + k * lattice_vectors[2]
                )
    translations = np.array(translations)  # (27, 3)

    # Replicate atoms
    supercell_coords = []
    supercell_types = []

    for trans in translations:
        supercell_coords.append(atom_coords + trans)
        supercell_types.extend(atom_types)

    supercell_coords = np.vstack(supercell_coords)  # (27*N, 3)

    # Compute distances from original atoms to all supercell atoms
    # We only need distances for the original N atoms.
    dists = cdist(atom_coords, supercell_coords)  # (N, 27*N)

    # Sort distances
    sorted_indices = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, sorted_indices, axis=1)

    # The first column is always 0 (self). We want neighbors.
    # Take columns 1 to k_max+1
    neighbor_dists = sorted_dists[:, 1 : k_max + 1]
    neighbor_indices_flat = sorted_indices[:, 1 : k_max + 1]

    neighbor_types = []
    for row_indices in neighbor_indices_flat:
        row_types = [supercell_types[idx] for idx in row_indices]
        neighbor_types.append(row_types)

    return neighbor_dists, neighbor_types


def get_atomic_features(lattice_vectors, atom_types, atom_coords):
    """
    Generates the atomic feature vector (21 dims).
    """
    n_atoms = len(atom_types)

    # 1. Atomic Identity (One-hot) -> 4 dims
    identity_feats = np.zeros((n_atoms, 4))
    for i, at in enumerate(atom_types):
        if at in ATOM_TO_INDEX:
            identity_feats[i, ATOM_TO_INDEX[at]] = 1.0

    # 2. Spatial Context (Centered Coords) -> 3 dims
    centroid = np.mean(atom_coords, axis=0)
    centered_coords = atom_coords - centroid

    # 3. Neighbor Search
    # Search for enough neighbors to cover K=24 and potential sparse species
    k_search = 32
    max_possible = 27 * n_atoms - 1
    k_search = min(k_search, max_possible)

    dists, n_types = compute_pbc_neighbors(
        lattice_vectors, atom_coords, atom_types, k_search
    )

    # 4. d_min -> 1 dim
    d_min = dists[:, 0:1]  # (N, 1)

    # 5. Packing Ratio -> 1 dim
    # Mean distance to 12 nearest neighbors
    k_packing = min(12, k_search)
    d_mean_12 = np.mean(dists[:, :k_packing], axis=1, keepdims=True)
    packing_ratio = d_min / (d_mean_12 + 1e-8)

    # 6. Multi-Scale Chemical Contexts -> 8 dims
    # K=6
    k6 = min(6, k_search)
    context_6 = np.zeros((n_atoms, 4))
    for i in range(n_atoms):
        local_dists = dists[i, :k6]
        local_types = n_types[i][:k6]
        weights = 1.0 / (local_dists + 1e-8)
        w_sum = np.sum(weights)
        for t, w in zip(local_types, weights):
            if t in ATOM_TO_INDEX:
                context_6[i, ATOM_TO_INDEX[t]] += w
        if w_sum > 1e-9:
            context_6[i] /= w_sum

    # K=24
    k24 = min(24, k_search)
    context_24 = np.zeros((n_atoms, 4))
    for i in range(n_atoms):
        local_dists = dists[i, :k24]
        local_types = n_types[i][:k24]
        weights = 1.0 / (local_dists + 1e-8)
        w_sum = np.sum(weights)
        for t, w in zip(local_types, weights):
            if t in ATOM_TO_INDEX:
                context_24[i, ATOM_TO_INDEX[t]] += w
        if w_sum > 1e-9:
            context_24[i] /= w_sum

    # 7. Chemically-Resolved Reciprocal Proximity -> 4 dims
    proximity = np.zeros((n_atoms, 4))
    for i in range(n_atoms):
        row_dists = dists[i]
        row_types = n_types[i]

        found = {t: False for t in ATOMIC_LABELS}
        for d, t in zip(row_dists, row_types):
            if t in ATOM_TO_INDEX and not found[t]:
                proximity[i, ATOM_TO_INDEX[t]] = 1.0 / (d + 1e-8)
                found[t] = True
            if all(found.values()):
                break

    # Concatenate all atomic features
    # Order: Identity(4), Coords(3), d_min(1), Packing(1), Context6(4), Context24(4), Proximity(4)
    features = np.hstack(
        [
            identity_feats,
            centered_coords,
            d_min,
            packing_ratio,
            context_6,
            context_24,
            proximity,
        ]
    )

    return features.astype(np.float32)


def get_global_features(lattice_vectors, atom_types):
    """
    Generates the global feature vector (21 dims).
    """
    # 1. Lattice Geometry -> 6 dims (3 lengths, 3 angles) + 1 Volume + 3 Aspect Ratios
    a = np.linalg.norm(lattice_vectors[0])
    b = np.linalg.norm(lattice_vectors[1])
    c = np.linalg.norm(lattice_vectors[2])

    # Angles
    alpha = np.degrees(
        np.arccos(
            np.clip(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c), -1.0, 1.0)
        )
    )
    beta = np.degrees(
        np.arccos(
            np.clip(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c), -1.0, 1.0)
        )
    )
    gamma = np.degrees(
        np.arccos(
            np.clip(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b), -1.0, 1.0)
        )
    )

    # Volume
    volume = np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )

    # Aspect Ratios
    lengths = sorted([a, b, c])
    ar1 = lengths[1] / (lengths[0] + 1e-8)
    ar2 = lengths[2] / (lengths[1] + 1e-8)
    ar3 = lengths[2] / (lengths[0] + 1e-8)

    # 2. Structural -> 2 dims
    n_atoms = len(atom_types)
    density = n_atoms / (volume + 1e-8)

    # 3. Chemical Stoichiometry -> 3 dims (Al, Ga, In percentages)
    counts = {t: 0 for t in ATOMIC_LABELS}
    for t in atom_types:
        if t in counts:
            counts[t] += 1

    pct_al = counts["Al"] / n_atoms
    pct_ga = counts["Ga"] / n_atoms
    pct_in = counts["In"] / n_atoms

    # 4. Physical Properties (Mean & Variance) -> 6 dims
    masses = [ATOMIC_MASSES.get(t, 0) for t in atom_types]
    radii = [COVALENT_RADII.get(t, 0) for t in atom_types]
    enegs = [ELECTRONEGATIVITY.get(t, 0) for t in atom_types]

    mean_mass = np.mean(masses)
    std_mass = np.std(masses)

    mean_radius = np.mean(radii)
    std_radius = np.std(radii)

    mean_eneg = np.mean(enegs)
    std_eneg = np.std(enegs)

    feat_vec = np.array(
        [
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            volume,
            ar1,
            ar2,
            ar3,
            density,
            float(n_atoms),
            pct_al,
            pct_ga,
            pct_in,
            mean_mass,
            mean_radius,
            mean_eneg,
            std_mass,
            std_radius,
            std_eneg,
        ],
        dtype=np.float32,
    )

    return feat_vec


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Processes the dataset: extracts features, scales them, and caches the result.
    """

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["atomic_features"],
                data["global_features"],
                data["targets"] if "targets" in data else None,
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    df = pd.read_csv(metadata_path)

    atomic_features_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    print(f"Processing {len(df)} samples from {metadata_path}...")

    for idx, row in df.iterrows():
        id_val = row["id"]
        file_path = row["file_path"]

        lattice, types, coords = parse_xyz(file_path)

        af = get_atomic_features(lattice, types, coords)
        gf = get_global_features(lattice, types)

        atomic_features_list.append(af)
        global_features_list.append(gf)
        ids_list.append(id_val)

        if "formation_energy_ev_natom" in row:
            t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
            targets_list.append(t)

    global_features = np.array(global_features_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=np.int32)

    if targets_list:
        targets = np.array(targets_list, dtype=np.float32)
        targets = np.log1p(targets)
    else:
        targets = None

    scaler_path = os.path.join(WORKING_DIR, "scalers.npz")
    is_train = "train.csv" in metadata_path

    all_atomic = np.vstack(atomic_features_list)

    if is_train:
        # Scale cols 4:21 (skip one-hot 0-3)
        af_mean = np.mean(all_atomic[:, 4:], axis=0)
        af_std = np.std(all_atomic[:, 4:], axis=0)
        af_std[af_std < 1e-8] = 1.0

        gf_mean = np.mean(global_features, axis=0)
        gf_std = np.std(global_features, axis=0)
        gf_std[gf_std < 1e-8] = 1.0

        np.savez(
            scaler_path, af_mean=af_mean, af_std=af_std, gf_mean=gf_mean, gf_std=gf_std
        )
        print("Scalers fitted and saved.")
    else:
        if os.path.exists(scaler_path):
            scalers = np.load(scaler_path)
            af_mean = scalers["af_mean"]
            af_std = scalers["af_std"]
            gf_mean = scalers["gf_mean"]
            gf_std = scalers["gf_std"]
        else:
            print("Warning: No scaler found. Skipping scaling.")
            af_mean = 0
            af_std = 1
            gf_mean = 0
            gf_std = 1

    all_atomic[:, 4:] = (all_atomic[:, 4:] - af_mean) / af_std

    current_idx = 0
    new_atomic_list = []
    for af in atomic_features_list:
        n = af.shape[0]
        new_atomic_list.append(all_atomic[current_idx : current_idx + n])
        current_idx += n
    atomic_features_list = np.array(new_atomic_list, dtype=object)

    global_features = (global_features - gf_mean) / gf_std
    global_features = global_features.astype(np.float32)

    if targets is not None:
        np.savez(
            cache_path,
            atomic_features=atomic_features_list,
            global_features=global_features,
            targets=targets,
            ids=ids,
        )
    else:
        np.savez(
            cache_path,
            atomic_features=atomic_features_list,
            global_features=global_features,
            ids=ids,
        )

    return atomic_features_list, global_features, targets, ids
