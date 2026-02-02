import os
import numpy as np
import pandas as pd
from library.config import Config


class GaussianRBF:
    """
    Gaussian Radial Basis Function expansion for edge/node features.
    Expands a scalar distance into a vector of RBF values.
    """

    def __init__(self, min_dist, max_dist, num_centers, gamma):
        self.centers = np.linspace(min_dist, max_dist, num_centers)
        self.gamma = gamma

    def expand(self, distances):
        """
        Expand scalar distances into RBF features.
        Args:
            distances: (N,) array of distances
        Returns:
            (N, num_centers) array of RBF features
        """
        # (N, 1) - (1, num_centers) -> (N, num_centers)
        return np.exp(-self.gamma * (distances[:, None] - self.centers[None, :]) ** 2)


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract atom types and positions.
    The file format contains lattice vectors and atom lines.
    We only parse lines starting with 'atom'.
    """
    atom_types = []
    positions = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "atom":
            # Format: atom x y z Type
            # Example: atom 1.67 7.51 6.55 Ga
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append([x, y, z])
                atom_types.append(parts[4])
            except (ValueError, IndexError):
                continue

    return atom_types, np.array(positions, dtype=np.float32)


def compute_centered_coords(positions):
    """
    Centers atomic coordinates by subtracting the geometric centroid.
    Args:
        positions: (N, 3) array of atomic coordinates
    Returns:
        centered_positions: (N, 3) array
    """
    if len(positions) == 0:
        return positions
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def get_atomic_features(atom_types, positions, rbf_expander):
    """
    Constructs the feature vector for each atom in the unit cell.
    Feature Vector Components:
    1. One-hot encoding of atom type (4 dims)
    2. Centered Cartesian coordinates (3 dims)
    3. RBF expansion of radial distance from centroid (32 dims)

    Args:
        atom_types: List of strings (e.g., ['Ga', 'Al', ...])
        positions: (N, 3) numpy array of coordinates
        rbf_expander: Instance of GaussianRBF

    Returns:
        features: (N, ATOMIC_INPUT_DIM) numpy array
    """
    num_atoms = len(atom_types)
    if num_atoms == 0:
        # Return dummy array of correct shape if no atoms found
        return np.zeros((0, Config.ATOMIC_INPUT_DIM), dtype=np.float32)

    # 1. One-hot encoding
    type_map = {t: i for i, t in enumerate(Config.ATOM_TYPES)}
    one_hot = np.zeros((num_atoms, Config.NUM_ATOM_TYPES), dtype=np.float32)

    for i, t in enumerate(atom_types):
        if t in type_map:
            one_hot[i, type_map[t]] = 1.0

    # 2. Centered Coordinates
    centered_pos = compute_centered_coords(positions)

    # 3. Radial Embedding
    # Calculate Euclidean distance of each atom from the centroid (0,0,0)
    distances = np.linalg.norm(centered_pos, axis=1)
    rbf_features = rbf_expander.expand(distances)

    # Concatenate all features
    # Shape: (N, 4 + 3 + 32) = (N, 39)
    features = np.concatenate([one_hot, centered_pos, rbf_features], axis=1)
    return features.astype(np.float32)


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Main processing function to prepare data for the model.
    Reads metadata, parses XYZ files, generates atomic and lattice features,
    transforms targets, and caches the results to disk.

    Args:
        metadata_path: Path to the metadata CSV file (train/val/test).
        cache_path: Path to save/load the .npz cache file.
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        Dictionary containing:
            - ids: (N,) array
            - atomic_features: (N,) object array of (M_i, D) arrays
            - lattice_features: (N, 7) array
            - targets: (N, 2) array
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "atomic_features": data["atomic_features"],
                "lattice_features": data["lattice_features"],
                "targets": data["targets"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from source...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Initialize RBF Expander
    rbf = GaussianRBF(
        min_dist=Config.RBF_MIN,
        max_dist=Config.RBF_MAX,
        num_centers=Config.NUM_RBF_CENTERS,
        gamma=Config.RBF_GAMMA,
    )

    # Lists to store processed data
    ids_list = []
    atomic_features_list = []
    lattice_features_list = []
    targets_list = []

    # Define columns to extract
    lattice_cols = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "number_of_total_atoms",
    ]

    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    has_targets = all(col in df.columns for col in target_cols)

    # Iterate over samples
    for _, row in df.iterrows():
        # ID
        ids_list.append(row["id"])

        # Lattice Features
        lat_feats = row[lattice_cols].values.astype(np.float32)
        lattice_features_list.append(lat_feats)

        # Targets
        if has_targets:
            t = row[target_cols].values.astype(np.float32)
            # Apply log(1+y) transformation for RMSLE metric compatibility
            # Ensure non-negative input for log1p (though physical energies should be valid)
            t_transformed = np.log1p(np.maximum(t, 0))
            targets_list.append(t_transformed)
        else:
            # Dummy targets for test set inference
            targets_list.append(np.zeros(2, dtype=np.float32))

        # Atomic Features
        # Construct full path to geometry file
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            atom_types, positions = parse_xyz(full_path)
            feats = get_atomic_features(atom_types, positions, rbf)
            atomic_features_list.append(feats)
        else:
            # Fallback (should not happen in valid dataset)
            print(f"Warning: File not found {full_path}")
            dummy_feats = np.zeros((1, Config.ATOMIC_INPUT_DIM), dtype=np.float32)
            atomic_features_list.append(dummy_feats)

    # Convert lists to numpy arrays
    # atomic_features is ragged (variable number of atoms), so we use object array
    atomic_features_arr = np.array(atomic_features_list, dtype=object)
    lattice_features_arr = np.array(lattice_features_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    ids_arr = np.array(ids_list, dtype=np.int32)

    # 3. Cache Data
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        ids=ids_arr,
        atomic_features=atomic_features_arr,
        lattice_features=lattice_features_arr,
        targets=targets_arr,
    )

    return {
        "ids": ids_arr,
        "atomic_features": atomic_features_arr,
        "lattice_features": lattice_features_arr,
        "targets": targets_arr,
    }
