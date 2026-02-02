import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.geometry_utils import (
    parse_xyz,
    get_pbc_distances,
    compute_chemical_densities,
)


class FeatureExtractor:
    """
    Extracts atomic and global features from material samples.
    """

    def __init__(self):
        self.atom_types = Config.ATOM_TYPES
        self.type_to_idx = {t: i for i, t in enumerate(self.atom_types)}

    def _get_one_hot(self, atom_type):
        """Returns one-hot encoding for an atom type."""
        one_hot = np.zeros(len(self.atom_types), dtype=np.float32)
        if atom_type in self.type_to_idx:
            one_hot[self.type_to_idx[atom_type]] = 1.0
        return one_hot

    def _get_lattice_params(self, lattice_vectors):
        """Calculates lattice lengths and angles from vectors."""
        a = np.linalg.norm(lattice_vectors[0])
        b = np.linalg.norm(lattice_vectors[1])
        c = np.linalg.norm(lattice_vectors[2])

        alpha = np.degrees(
            np.arccos(np.dot(lattice_vectors[1], lattice_vectors[2]) / (b * c))
        )
        beta = np.degrees(
            np.arccos(np.dot(lattice_vectors[0], lattice_vectors[2]) / (a * c))
        )
        gamma = np.degrees(
            np.arccos(np.dot(lattice_vectors[0], lattice_vectors[1]) / (a * b))
        )

        return np.array([a, b, c]), np.array([alpha, beta, gamma])

    def _get_volume(self, lattice_vectors):
        """Calculates unit cell volume."""
        return np.abs(
            np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
        )

    def process_sample(self, file_path):
        """
        Process a single geometry file to extract features.

        Returns:
            atomic_features (np.ndarray): (N, 12)
            global_features (np.ndarray): (12,)
        """
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        lattice_vectors, atom_coords, atom_types_list = parse_xyz(full_path)

        num_atoms = len(atom_coords)

        # --- Atomic Features ---

        # 1. One-hot encoding (N, 4)
        one_hots = np.array([self._get_one_hot(t) for t in atom_types_list])

        # 2. Centered Coordinates (N, 3)
        # Calculate centroid of the unit cell atoms
        centroid = np.mean(atom_coords, axis=0)
        centered_coords = atom_coords - centroid

        # 3. Nearest Neighbor Distance (N, 1)
        # Compute PBC distances
        dists = get_pbc_distances(atom_coords, lattice_vectors)
        # Mask self-distance (diagonal is 0)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1).reshape(-1, 1)

        # 4. Chemical Density Fields (N, 4)
        densities = compute_chemical_densities(
            atom_coords,
            atom_types_list,
            lattice_vectors,
            self.atom_types,
            Config.DENSITY_GAMMA,
            Config.DENSITY_CUTOFF,
        )

        # Concatenate atomic features: 4 + 3 + 1 + 4 = 12
        atomic_features = np.hstack([one_hots, centered_coords, nn_dists, densities])

        # --- Global Features ---

        # Lattice parameters
        lengths, angles = self._get_lattice_params(lattice_vectors)

        # Volume
        volume = self._get_volume(lattice_vectors)

        # Atomic Density
        density = num_atoms / volume

        # Stoichiometry (Al, Ga, In fractions)
        # Config.ATOM_TYPES = ["Al", "Ga", "In", "O"]
        counts = {t: 0 for t in self.atom_types}
        for t in atom_types_list:
            if t in counts:
                counts[t] += 1

        stoich = np.array(
            [
                counts["Al"] / num_atoms,
                counts["Ga"] / num_atoms,
                counts["In"] / num_atoms,
            ]
        )

        # Concatenate global features: 3 + 3 + 1 + 1 + 1 + 3 = 12
        # lengths, angles, volume, density, total_atoms, stoich
        global_features = np.concatenate(
            [lengths, angles, [volume], [density], [float(num_atoms)], stoich]
        )

        return atomic_features.astype(np.float32), global_features.astype(np.float32)


class PreprocessPipeline:
    """
    Handles scaling of features and transformation of targets.
    """

    def __init__(self):
        self.atomic_mean = None
        self.atomic_scale = None
        self.global_mean = None
        self.global_scale = None

    def fit(self, atomic_feats, global_feats):
        """Compute mean and std for scaling."""
        # atomic_feats is (Total_N, D_atomic)
        self.atomic_mean = np.mean(atomic_feats, axis=0)
        self.atomic_scale = np.std(atomic_feats, axis=0)
        # Avoid division by zero
        self.atomic_scale[self.atomic_scale == 0] = 1.0

        # global_feats is (M, D_global)
        self.global_mean = np.mean(global_feats, axis=0)
        self.global_scale = np.std(global_feats, axis=0)
        self.global_scale[self.global_scale == 0] = 1.0

    def transform(self, atomic_feats, global_feats):
        """Apply standard scaling."""
        if self.atomic_mean is None or self.global_mean is None:
            raise ValueError("Pipeline must be fitted before transform.")

        scaled_atomic = (atomic_feats - self.atomic_mean) / self.atomic_scale
        scaled_global = (global_feats - self.global_mean) / self.global_scale

        return scaled_atomic.astype(np.float32), scaled_global.astype(np.float32)

    def transform_targets(self, targets):
        """Apply log(1+x) transformation to targets."""
        return np.log1p(targets).astype(np.float32)

    def inverse_transform_targets(self, transformed_targets):
        """Apply exp(x)-1 transformation to predictions."""
        return np.expm1(transformed_targets)

    def save_scalers(self, path):
        """Save scaler parameters to npz."""
        np.savez(
            path,
            atomic_mean=self.atomic_mean,
            atomic_scale=self.atomic_scale,
            global_mean=self.global_mean,
            global_scale=self.global_scale,
        )

    def load_scalers(self, path):
        """Load scaler parameters from npz."""
        data = np.load(path)
        self.atomic_mean = data["atomic_mean"]
        self.atomic_scale = data["atomic_scale"]
        self.global_mean = data["global_mean"]
        self.global_scale = data["global_scale"]


def process_dataset(metadata_path, load_cached_data=True, is_test=False):
    """
    Main function to process a dataset (train, val, or test).

    Args:
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to try loading from cache.
        is_test (bool): Whether this is the test set (no targets).

    Returns:
        dict containing:
            'atomic_features': Flattened atomic features (Total_N, D_atomic)
            'sample_indices': Index mapping atoms to samples (Total_N,)
            'global_features': Global features (M, D_global)
            'targets': Targets (M, 2) or None
            'ids': Sample IDs (M,)
    """
    # Determine cache file name based on input path
    if "train.csv" in metadata_path:
        cache_name = Config.TRAIN_CACHE_FILE
    elif "val.csv" in metadata_path:
        cache_name = Config.VAL_CACHE_FILE
    elif "test.csv" in metadata_path:
        cache_name = Config.TEST_CACHE_FILE
    else:
        cache_name = "custom_data.npz"

    cache_path = os.path.join(Config.WORKING_DIR, cache_name)
    Config.make_dirs()

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            result = {
                "atomic_features": data["atomic_features"],
                "sample_indices": data["sample_indices"],
                "global_features": data["global_features"],
                "ids": data["ids"],
            }
            if not is_test:
                result["targets"] = data["targets"]
            else:
                result["targets"] = None
            return result
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    extractor = FeatureExtractor()

    all_atomic_feats = []
    all_sample_indices = []
    all_global_feats = []
    all_targets = []
    all_ids = []

    # Iterate over samples
    for idx, row in df.iterrows():
        # Extract features
        af, gf = extractor.process_sample(row["file_path"])

        # Store features
        all_atomic_feats.append(af)
        # Create indices mapping these atoms to the current sample index (0 to M-1)
        # We use the loop index `idx` which corresponds to the row in the dataframe/arrays
        # But to be safe and contiguous, let's use the length of the lists so far
        sample_idx = len(all_ids)
        all_sample_indices.append(np.full(af.shape[0], sample_idx, dtype=np.int32))

        all_global_feats.append(gf)
        all_ids.append(row["id"])

        if not is_test:
            targets = np.array(
                [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]],
                dtype=np.float32,
            )
            all_targets.append(targets)

    # Concatenate
    atomic_features_flat = np.vstack(all_atomic_feats)
    sample_indices_flat = np.concatenate(all_sample_indices)
    global_features_arr = np.vstack(all_global_feats)
    ids_arr = np.array(all_ids, dtype=np.int32)

    targets_arr = np.vstack(all_targets) if not is_test else np.array([])

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    save_dict = {
        "atomic_features": atomic_features_flat,
        "sample_indices": sample_indices_flat,
        "global_features": global_features_arr,
        "ids": ids_arr,
    }
    if not is_test:
        save_dict["targets"] = targets_arr

    np.savez(cache_path, **save_dict)

    # Return result dict
    result = {
        "atomic_features": atomic_features_flat,
        "sample_indices": sample_indices_flat,
        "global_features": global_features_arr,
        "ids": ids_arr,
        "targets": targets_arr if not is_test else None,
    }

    return result
