import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALERS_CACHE_PATH,
    ATOM_INPUT_DIM,
    GLOBAL_INPUT_DIM,
)
from library.geometry_utils import process_geometry

SPECIES = ["Al", "Ga", "In", "O"]
SPECIES_MAP = {s: i for i, s in enumerate(SPECIES)}


def get_one_hot(species_list):
    """
    Converts a list of species strings to a one-hot numpy array.
    """
    n = len(species_list)
    one_hot = np.zeros((n, 4), dtype=np.float32)
    for i, s in enumerate(species_list):
        if s in SPECIES_MAP:
            one_hot[i, SPECIES_MAP[s]] = 1.0
    return one_hot


def get_atomic_features(geometry_data):
    """
    Constructs the 12-dimensional atomic feature vectors.

    Features:
    1. Self Identity (One-hot, 4 dims)
    2. Spatial Context (Centered Coords, 3 dims)
    3. Nearest Neighbor Distance (Scalar, 1 dim)
    4. Nearest Neighbor Identity (One-hot, 4 dims)
    """
    species = geometry_data["species"]
    coords = geometry_data["coords"]  # Centered
    nn_dist = geometry_data["nn_dist"]
    nn_species = geometry_data["nn_species"]

    n_atoms = len(species)

    # 1. Self Identity
    self_oh = get_one_hot(species)

    # 2. Spatial Context (already centered in geometry_data)
    spatial = coords.astype(np.float32)

    # 3. NN Distance
    dist = nn_dist.reshape(-1, 1).astype(np.float32)

    # 4. NN Identity
    nn_oh = get_one_hot(nn_species)

    # Concatenate
    # Order: Self(4) | Spatial(3) | Dist(1) | NN(4)
    # Indices: 0-3 | 4-6 | 7 | 8-11
    features = np.hstack([self_oh, spatial, dist, nn_oh])

    return features


def calculate_volume(lattice_vectors):
    """Calculates the volume of the unit cell."""
    return np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )


def get_global_features(row, geometry_data):
    """
    Constructs the 12-dimensional global feature vector.

    Features:
    1. Lattice lengths (3 dims)
    2. Lattice angles (3 dims)
    3. Volume (1 dim)
    4. Atomic Density (1 dim)
    5. Stoichiometry (3 dims: Al, Ga, In)
    6. Total Atoms (1 dim)
    """
    # 1. Lattice lengths
    lengths = np.array(
        [
            row["lattice_vector_1_ang"],
            row["lattice_vector_2_ang"],
            row["lattice_vector_3_ang"],
        ],
        dtype=np.float32,
    )

    # 2. Lattice angles
    angles = np.array(
        [
            row["lattice_angle_alpha_degree"],
            row["lattice_angle_beta_degree"],
            row["lattice_angle_gamma_degree"],
        ],
        dtype=np.float32,
    )

    # 3. Volume
    # We can compute it from lattice vectors in geometry_data or use the formula with lengths/angles.
    # Using geometry_data is more direct.
    vol = calculate_volume(geometry_data["lattice"])

    # 4. Atomic Density
    n_atoms = row["number_of_total_atoms"]
    density = n_atoms / vol

    # 5. Stoichiometry
    stoich = np.array(
        [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]],
        dtype=np.float32,
    )

    # 6. Total Atoms
    total_atoms = np.array([n_atoms], dtype=np.float32)

    # Concatenate
    features = np.concatenate(
        [
            lengths,
            angles,
            np.array([vol], dtype=np.float32),
            np.array([density], dtype=np.float32),
            stoich,
            total_atoms,
        ]
    )

    return features


def process_dataset(
    metadata_path,
    cache_path,
    load_cached_data=True,
    fit_scalers=False,
    scalers_path=SCALERS_CACHE_PATH,
):
    """
    Processes the dataset: extracts features, scales them, and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to save/load the .npz cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        fit_scalers (bool): Whether to fit new scalers (True for train, False for val/test).
        scalers_path (str): Path to save/load scalers.

    Returns:
        dict: containing 'atomic_features', 'global_features', 'targets', 'ids'
    """

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "atomic_features": data["atomic_features"],  # object array of arrays
                "global_features": data["global_features"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing dataset from {metadata_path}")
    df = pd.read_csv(metadata_path)

    all_atomic_features = []  # List of (N, 12) arrays
    all_global_features = []  # List of (12,) arrays
    all_targets = []
    all_ids = []

    for idx, row in df.iterrows():
        # Path to geometry file
        # metadata contains relative path, e.g. "train/1/geometry.xyz"
        # INPUT_DIR is "./input"
        geo_path = os.path.join(INPUT_DIR, row["file_path"])

        # Parse and process geometry
        geo_data = process_geometry(geo_path)

        # Extract features
        af = get_atomic_features(geo_data)
        gf = get_global_features(row, geo_data)

        all_atomic_features.append(af)
        all_global_features.append(gf)
        all_ids.append(row["id"])

        # Extract targets if available
        if "formation_energy_ev_natom" in row:
            # Log transform targets: log(1 + y)
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            all_targets.append([t1, t2])
        else:
            # Placeholder for test set
            all_targets.append([0.0, 0.0])

    # Convert global to array
    all_global_features = np.array(all_global_features, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)
    all_ids = np.array(all_ids, dtype=np.int32)

    # 3. Scaling
    # Atomic Continuous Indices: 4, 5, 6 (coords), 7 (dist)
    # Global Continuous Indices: All 0-11

    atomic_cont_indices = [4, 5, 6, 7]

    # Collect all atomic continuous data for fitting
    if fit_scalers:
        print("Fitting scalers...")
        # Stack all atomic features to fit scaler on continuous columns
        # We need to be careful not to copy everything if memory is tight, but dataset is small (~2k samples)
        flat_atomic = np.vstack(all_atomic_features)
        atomic_cont_data = flat_atomic[:, atomic_cont_indices]

        scaler_atomic = StandardScaler()
        scaler_atomic.fit(atomic_cont_data)

        scaler_global = StandardScaler()
        scaler_global.fit(all_global_features)

        # Save scalers parameters manually to npz to avoid pickle
        np.savez(
            scalers_path,
            atomic_mean=scaler_atomic.mean_,
            atomic_scale=scaler_atomic.scale_,
            global_mean=scaler_global.mean_,
            global_scale=scaler_global.scale_,
        )
    else:
        print(f"Loading scalers from {scalers_path}")
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(
                f"Scalers file not found at {scalers_path}. Run training first."
            )

        scalers_data = np.load(scalers_path)

        # Reconstruct scalers (or just apply math directly)
        # We'll apply math directly for simplicity and robustness
        atomic_mean = scalers_data["atomic_mean"]
        atomic_scale = scalers_data["atomic_scale"]
        global_mean = scalers_data["global_mean"]
        global_scale = scalers_data["global_scale"]

    # Apply Scaling
    # Atomic
    if fit_scalers:
        atomic_mean = scaler_atomic.mean_
        atomic_scale = scaler_atomic.scale_
        global_mean = scaler_global.mean_
        global_scale = scaler_global.scale_

    for i in range(len(all_atomic_features)):
        # (x - mean) / scale
        all_atomic_features[i][:, atomic_cont_indices] = (
            all_atomic_features[i][:, atomic_cont_indices] - atomic_mean
        ) / atomic_scale

    # Global
    all_global_features = (all_global_features - global_mean) / global_scale

    # 4. Save to cache
    # We save atomic_features as an object array because sub-arrays have different lengths (N_atoms)
    # Using allow_pickle=True is necessary for object arrays in np.savez, but we are storing numpy arrays inside, not arbitrary python objects.
    # This is standard for jagged arrays in numpy.
    print(f"Saving processed data to {cache_path}")

    # Create object array for atomic features
    atomic_features_obj = np.empty(len(all_atomic_features), dtype=object)
    for i, arr in enumerate(all_atomic_features):
        atomic_features_obj[i] = arr

    np.savez(
        cache_path,
        atomic_features=atomic_features_obj,
        global_features=all_global_features,
        targets=all_targets,
        ids=all_ids,
    )

    return {
        "atomic_features": atomic_features_obj,
        "global_features": all_global_features,
        "targets": all_targets,
        "ids": all_ids,
    }
