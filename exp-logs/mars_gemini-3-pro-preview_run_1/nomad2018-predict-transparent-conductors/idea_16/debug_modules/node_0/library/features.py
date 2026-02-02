import os
import numpy as np
import pandas as pd
import torch
from library import config
from library import geometry


class DataScaler:
    """
    Manages scaling of continuous features using standardization (mean/std).
    """

    def __init__(self):
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None
        self.fitted = False

    def fit(self, atomic_features_list, global_features):
        """
        Compute mean and std for atomic and global features.

        Args:
            atomic_features_list (list of np.ndarray): List of atomic feature arrays.
                                                       Each array is (N_atoms, Feature_Dim).
                                                       Indices 4: (Cartesian, Fractional, NN, Pot) are continuous.
                                                       Indices 0-3 are One-Hot (don't scale).
            global_features (np.ndarray): Array of global features (B, Global_Dim).
        """
        # Atomic features: Concatenate all atoms from all crystals
        # Features: [OneHot(4), Cart(3), Frac(3), NN(1), Pot(1)] -> Total 12
        # Continuous indices: 4 to 11
        all_atomic = np.concatenate(atomic_features_list, axis=0)

        # We only scale the continuous parts
        continuous_atomic = all_atomic[:, 4:]
        self.atomic_mean = np.mean(continuous_atomic, axis=0)
        self.atomic_std = np.std(continuous_atomic, axis=0)
        # Avoid division by zero
        self.atomic_std[self.atomic_std < 1e-8] = 1.0

        # Global features
        self.global_mean = np.mean(global_features, axis=0)
        self.global_std = np.std(global_features, axis=0)
        self.global_std[self.global_std < 1e-8] = 1.0

        self.fitted = True

    def transform(self, atomic_features_list, global_features):
        """
        Apply scaling to features.
        """
        if not self.fitted:
            raise RuntimeError("DataScaler must be fitted before transform.")

        scaled_atomic_list = []
        for feat in atomic_features_list:
            feat_copy = feat.copy()
            # Scale continuous columns (4:)
            feat_copy[:, 4:] = (feat_copy[:, 4:] - self.atomic_mean) / self.atomic_std
            scaled_atomic_list.append(feat_copy)

        scaled_global = (global_features - self.global_mean) / self.global_std

        return scaled_atomic_list, scaled_global

    def save(self, path):
        np.savez(
            path,
            atomic_mean=self.atomic_mean,
            atomic_std=self.atomic_std,
            global_mean=self.global_mean,
            global_std=self.global_std,
            fitted=np.array([self.fitted]),
        )

    def load(self, path):
        data = np.load(path)
        self.atomic_mean = data["atomic_mean"]
        self.atomic_std = data["atomic_std"]
        self.global_mean = data["global_mean"]
        self.global_std = data["global_std"]
        self.fitted = bool(data["fitted"][0])


class MaterialFeatureExtractor:
    """
    Handles loading, feature extraction, and caching of material data.
    """

    def __init__(self):
        pass

    def _get_one_hot(self, atom_types):
        """
        Convert list of atom symbols to one-hot encoding.
        """
        # Map: Al:0, Ga:1, In:2, O:3
        mapping = config.ATOM_MAP
        num_types = config.NUM_ATOM_TYPES
        one_hot = np.zeros((len(atom_types), num_types), dtype=np.float32)
        for i, sym in enumerate(atom_types):
            if sym in mapping:
                one_hot[i, mapping[sym]] = 1.0
        return one_hot

    def process_data(self, df, split_name, load_cached_data=True, scaler=None):
        """
        Main pipeline to process data.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.
            scaler (DataScaler): Scaler instance. Required if split_name is 'val' or 'test'.
                                 If 'train', it will be fitted.

        Returns:
            dict: {
                'atomic_features': list of np.ndarray,
                'global_features': np.ndarray,
                'targets': np.ndarray,
                'ids': np.ndarray
            }
        """
        cache_file = os.path.join(config.WORKING_DIR, f"{split_name}_data.npz")

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split_name} data from cache: {cache_file}")
            try:
                data = np.load(cache_file, allow_pickle=True)
                # atomic_features is saved as an object array of arrays
                atomic_features = list(data["atomic_features"])
                global_features = data["global_features"]
                targets = data["targets"]
                ids = data["ids"]
                return {
                    "atomic_features": atomic_features,
                    "global_features": global_features,
                    "targets": targets,
                    "ids": ids,
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Processing {split_name} data from scratch...")

        atomic_features_list = []
        global_features_list = []
        targets_list = []
        ids_list = []

        # Loop through dataframe
        for idx, row in df.iterrows():
            mat_id = row["id"]

            # --- Geometry Processing ---
            # Construct full path to geometry file
            # The metadata file_path is relative to input dir
            xyz_path = os.path.join(config.INPUT_DIR, row["file_path"])

            if not os.path.exists(xyz_path):
                print(f"Warning: File not found {xyz_path}, skipping.")
                continue

            # Read XYZ
            lattice_matrix, atom_types, cart_coords = geometry.read_xyz(xyz_path)

            # 1. Atomic Identity (One-Hot)
            one_hot = self._get_one_hot(atom_types)

            # 2. Centered Cartesian Coordinates
            centroid = np.mean(cart_coords, axis=0)
            centered_cart = cart_coords - centroid

            # 3. Centered Fractional Coordinates
            frac_coords = geometry.cartesian_to_fractional(cart_coords, lattice_matrix)
            frac_centroid = np.mean(frac_coords, axis=0)
            centered_frac = frac_coords - frac_centroid

            # 4. PBC Distances & Potential
            # Use fractional coords for PBC distance calculation
            dist_matrix = geometry.get_pbc_distances(frac_coords, lattice_matrix)
            potential, nn_dist = geometry.compute_local_potential(dist_matrix)

            # Reshape for concatenation
            nn_dist = nn_dist[:, np.newaxis]
            potential = potential[:, np.newaxis]

            # Assemble Atomic Feature Vector
            # [OneHot(4), Cart(3), Frac(3), NN(1), Pot(1)] -> 12 dims
            atom_feat = np.concatenate(
                [one_hot, centered_cart, centered_frac, nn_dist, potential], axis=1
            ).astype(np.float32)

            atomic_features_list.append(atom_feat)

            # --- Global Processing ---
            # Extract from DF
            # Lattice Lengths (3)
            lat_lens = np.array(
                [
                    row["lattice_vector_1_ang"],
                    row["lattice_vector_2_ang"],
                    row["lattice_vector_3_ang"],
                ]
            )
            # Lattice Angles (3)
            lat_angs = np.array(
                [
                    row["lattice_angle_alpha_degree"],
                    row["lattice_angle_beta_degree"],
                    row["lattice_angle_gamma_degree"],
                ]
            )

            # Volume Calculation (using geometry module logic or simple formula)
            # We can compute volume from the lattice matrix extracted from XYZ for consistency
            # Vol = det(matrix)
            vol = np.abs(np.linalg.det(lattice_matrix))

            # Total Atoms
            n_atoms = len(atom_types)

            # Atomic Density
            density = n_atoms / vol

            # Stoichiometry (3) - Al, Ga, In
            stoich = np.array(
                [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
            )

            # Assemble Global Feature Vector
            # [Lens(3), Angs(3), Vol(1), Dens(1), Stoich(3), N_Atoms(1)] -> 12 dims
            glob_feat = np.concatenate(
                [lat_lens, lat_angs, [vol], [density], stoich, [n_atoms]]
            ).astype(np.float32)

            global_features_list.append(glob_feat)

            # --- Targets ---
            if split_name == "test":
                # Dummy targets for test set
                targets_list.append(np.zeros(2, dtype=np.float32))
            else:
                # Log transform targets: log(1 + y)
                t1 = np.log1p(row["formation_energy_ev_natom"])
                t2 = np.log1p(row["bandgap_energy_ev"])
                targets_list.append(np.array([t1, t2], dtype=np.float32))

            ids_list.append(mat_id)

        # Convert lists to arrays
        # atomic_features_list remains a list of arrays because N_atoms varies
        global_features = np.array(global_features_list, dtype=np.float32)
        targets = np.array(targets_list, dtype=np.float32)
        ids = np.array(ids_list, dtype=np.int32)

        # --- Scaling ---
        if scaler is None:
            # Should happen for training set if not provided
            scaler = DataScaler()

        if split_name == "train" and not scaler.fitted:
            print("Fitting scaler on training data...")
            scaler.fit(atomic_features_list, global_features)
            scaler.save(config.SCALERS_CACHE)
        elif not scaler.fitted:
            # If we are in val/test but scaler isn't fitted, try to load
            if os.path.exists(config.SCALERS_CACHE):
                print("Loading scaler from cache...")
                scaler.load(config.SCALERS_CACHE)
            else:
                raise RuntimeError(
                    "Scaler not fitted and no cache found. Run training first."
                )

        # Apply transform
        scaled_atomic, scaled_global = scaler.transform(
            atomic_features_list, global_features
        )

        # --- Save to Cache ---
        # Note: np.savez handles lists of arrays by pickling them inside the zip if we use object array
        # To be safe and cleaner, we save atomic features as object array
        atomic_obj_arr = np.array(scaled_atomic, dtype=object)

        np.savez(
            cache_file,
            atomic_features=atomic_obj_arr,
            global_features=scaled_global,
            targets=targets,
            ids=ids,
        )
        print(f"Saved {split_name} data to cache.")

        return {
            "atomic_features": scaled_atomic,
            "global_features": scaled_global,
            "targets": targets,
            "ids": ids,
        }
