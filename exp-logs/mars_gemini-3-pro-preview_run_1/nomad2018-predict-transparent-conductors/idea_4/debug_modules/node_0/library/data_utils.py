import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from library.config import Config


def parse_xyz(file_path):
    """
    Parses an xyz file to extract atomic coordinates and types.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    coords = []
    types = []

    with open(full_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Lines starting with 'atom' contain coordinate info
        if parts[0] == "atom":
            # Format: atom x y z type
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            types.append(parts[4])

    return np.array(coords), np.array(types)


def center_coordinates(coords):
    """
    Centers the atomic coordinates relative to the unit cell centroid.
    """
    if len(coords) == 0:
        return coords
    centroid = np.mean(coords, axis=0)
    return coords - centroid


def compute_nearest_neighbor_distance(coords):
    """
    Computes the distance to the nearest neighbor for each atom.
    Returns a (N, 1) array.
    """
    n_atoms = len(coords)
    if n_atoms <= 1:
        return np.zeros((n_atoms, 1))

    # Compute pairwise Euclidean distances
    dists = squareform(pdist(coords))

    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)

    # Find min distance for each atom
    min_dists = np.min(dists, axis=1).reshape(-1, 1)
    return min_dists


def get_atomic_features(file_path):
    """
    Constructs the feature vector for the atomic stream.
    Features: [One-hot(4), x, y, z, nn_dist] -> Total 8 dims
    """
    coords, types = parse_xyz(file_path)

    # 1. One-hot encoding of element type (Al, Ga, In, O)
    # Map: Al->0, Ga->1, In->2, O->3
    type_map = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    one_hot = np.zeros((len(types), 4))
    for i, t in enumerate(types):
        if t in type_map:
            one_hot[i, type_map[t]] = 1.0

    # 2. Centered Cartesian coordinates
    centered_coords = center_coordinates(coords)

    # 3. Nearest neighbor distance
    nn_dist = compute_nearest_neighbor_distance(coords)

    # Concatenate features
    features = np.hstack([one_hot, centered_coords, nn_dist])
    return features


def calculate_volume(a, b, c, alpha, beta, gamma):
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

    volume = a * b * c * np.sqrt(np.maximum(0, term))
    return volume


def process_tabular_features(row):
    """
    Extracts and computes global features from a dataframe row.
    """
    # Extract basic features
    lv1 = row["lattice_vector_1_ang"]
    lv2 = row["lattice_vector_2_ang"]
    lv3 = row["lattice_vector_3_ang"]
    alpha = row["lattice_angle_alpha_degree"]
    beta = row["lattice_angle_beta_degree"]
    gamma = row["lattice_angle_gamma_degree"]
    n_atoms = row["number_of_total_atoms"]
    p_al = row["percent_atom_al"]
    p_ga = row["percent_atom_ga"]
    p_in = row["percent_atom_in"]

    # Compute derived features
    volume = calculate_volume(lv1, lv2, lv3, alpha, beta, gamma)
    density = n_atoms / volume if volume > 0 else 0

    # Feature vector construction (12 dimensions)
    features = np.array(
        [lv1, lv2, lv3, alpha, beta, gamma, n_atoms, p_al, p_ga, p_in, volume, density]
    )

    return features


def prepare_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads data from metadata CSV, processes features, and caches the result.
    Applies log(1+x) transformation to targets.
    """
    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "global_features": data["global_features"],
                "atomic_features": data["atomic_features"],
                "targets": data["targets"] if "targets" in data else None,
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    ids = df["id"].values
    global_features_list = []
    atomic_features_list = []
    targets_list = []

    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    for idx, row in df.iterrows():
        # Global features
        g_feat = process_tabular_features(row)
        global_features_list.append(g_feat)

        # Atomic features
        a_feat = get_atomic_features(row["file_path"])
        atomic_features_list.append(a_feat)

        # Targets
        if has_targets:
            # Apply log(1+y) transformation to align with RMSLE metric
            t = row[Config.TARGET_COLS].values.astype(float)
            t_log = np.log1p(t)
            targets_list.append(t_log)

    global_features = np.array(global_features_list, dtype=np.float32)
    # Atomic features are variable length, keep as object array of arrays
    atomic_features = np.array(atomic_features_list, dtype=object)

    targets = None
    if has_targets:
        targets = np.array(targets_list, dtype=np.float32)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {
        "ids": ids,
        "global_features": global_features,
        "atomic_features": atomic_features,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Data saved to {cache_path}")

    return {
        "ids": ids,
        "global_features": global_features,
        "atomic_features": atomic_features,
        "targets": targets,
    }


def normalize_data(train_data, val_data, test_data):
    """
    Calculates mean and std from training data and normalizes all splits.
    Scales global features and continuous atomic features (coords, nn_dist).
    """
    # 1. Global Features Normalization
    mean_g = np.mean(train_data["global_features"], axis=0)
    std_g = np.std(train_data["global_features"], axis=0)
    # Avoid division by zero
    std_g[std_g == 0] = 1.0

    train_data["global_features"] = (train_data["global_features"] - mean_g) / std_g
    val_data["global_features"] = (val_data["global_features"] - mean_g) / std_g
    test_data["global_features"] = (test_data["global_features"] - mean_g) / std_g

    # 2. Atomic Features Normalization
    # Features: [One-hot(4), x, y, z, dist]
    # Indices to scale: 4, 5, 6 (coords), 7 (dist)
    indices_to_scale = [4, 5, 6, 7]

    # Collect all atomic vectors from training set to compute stats
    all_train_atoms = np.concatenate(train_data["atomic_features"], axis=0)

    mean_a = np.mean(all_train_atoms[:, indices_to_scale], axis=0)
    std_a = np.std(all_train_atoms[:, indices_to_scale], axis=0)
    std_a[std_a == 0] = 1.0

    def apply_atom_norm(data_dict):
        normalized_atoms = []
        for atom_arr in data_dict["atomic_features"]:
            new_arr = atom_arr.copy()
            new_arr[:, indices_to_scale] = (
                new_arr[:, indices_to_scale] - mean_a
            ) / std_a
            normalized_atoms.append(new_arr)
        data_dict["atomic_features"] = np.array(normalized_atoms, dtype=object)

    apply_atom_norm(train_data)
    apply_atom_norm(val_data)
    apply_atom_norm(test_data)

    return train_data, val_data, test_data


def load_and_preprocess_data():
    """
    Main function to load all splits and apply normalization.
    """
    # Load raw (or cached raw) data
    train_data = prepare_dataset(Config.TRAIN_CSV, Config.TRAIN_CACHE)
    val_data = prepare_dataset(Config.VAL_CSV, Config.VAL_CACHE)
    test_data = prepare_dataset(Config.TEST_CSV, Config.TEST_CACHE)

    # Apply normalization based on training statistics
    train_data, val_data, test_data = normalize_data(train_data, val_data, test_data)

    return train_data, val_data, test_data
