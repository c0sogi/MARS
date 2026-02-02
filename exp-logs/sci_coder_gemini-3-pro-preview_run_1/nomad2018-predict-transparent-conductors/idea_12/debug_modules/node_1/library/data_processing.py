import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import get_logger

logger = get_logger("data_processing")


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material data.
    """

    def __init__(self, atomic_features_list, global_features, targets=None, ids=None):
        """
        Args:
            atomic_features_list (list of np.ndarray): List where each element is (N_atoms, Feature_Dim).
            global_features (np.ndarray): Array of shape (N_samples, Global_Dim).
            targets (np.ndarray, optional): Array of shape (N_samples, Target_Dim).
            ids (np.ndarray, optional): Array of IDs.
        """
        self.atomic_features_list = atomic_features_list
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.atomic_features_list)

    def __getitem__(self, idx):
        item = {
            "atomic_features": torch.tensor(
                self.atomic_features_list[idx], dtype=torch.float32
            ),
            "global_features": torch.tensor(
                self.global_features[idx], dtype=torch.float32
            ),
            "id": int(self.ids[idx]) if self.ids is not None else -1,
        }
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class DataHandler:
    """
    Handles data loading, feature engineering, scaling, and caching.
    """

    def __init__(self):
        # Scalers for continuous features
        # Atomic continuous: Coords (3) + NN Dist (1) + Potential (1) = 5
        self.atomic_scaler = StandardScaler()
        # Global: Lattice(3+3) + Vol(1) + Dens(1) + Comp(3) = 11
        self.global_scaler = StandardScaler()
        self.is_fitted = False

    def parse_xyz(self, file_path):
        """
        Parses a geometry.xyz file.
        Returns:
            lattice_vectors (np.ndarray): 3x3 matrix
            atoms (list of tuples): [(atom_type, x, y, z), ...]
        """
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        lattice_vectors = []
        atoms = []

        with open(full_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "lattice_vector":
                    lattice_vectors.append([float(x) for x in parts[1:4]])
                elif parts[0] == "atom":
                    # Format: atom x y z type
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atom_type = parts[4]
                    atoms.append((atom_type, x, y, z))

        return np.array(lattice_vectors), atoms

    def compute_pbc_distances(self, coords, lattice):
        """
        Computes pairwise distances considering Periodic Boundary Conditions.
        Uses a 3x3x3 supercell approach for robustness with small unit cells.

        Args:
            coords (np.ndarray): (N, 3) atomic coordinates.
            lattice (np.ndarray): (3, 3) lattice vectors.

        Returns:
            dist_matrix (np.ndarray): (N, N) minimum distance matrix (not strictly needed for this feature set,
                                      but we need NN dist and potential).
            all_dists (list of np.arrays): For each atom, list of distances to all neighbors in supercell.
        """
        n_atoms = len(coords)

        # Generate supercell offsets (-1, 0, 1)
        offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    offsets.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
        offsets = np.array(offsets)  # (27, 3)

        # Expand coords to supercell: (27*N, 3)
        # However, we only need distances from the original unit cell atoms to the supercell atoms.

        # Vectorized calculation
        # For each atom i in original cell, calculate dist to all atoms j in all 27 images.

        # This can be memory intensive if vectorized fully (N * 27N).
        # Given N ~ 80, 80 * 2160 = 172k pairs. This is small.

        super_coords = []
        for off in offsets:
            super_coords.append(coords + off)
        super_coords = np.vstack(super_coords)  # (27*N, 3)

        # Compute distances from original coords to super_coords
        # coords: (N, 1, 3)
        # super_coords: (1, 27N, 3)
        # diff: (N, 27N, 3)
        diff = coords[:, np.newaxis, :] - super_coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, 27N)

        return dists

    def extract_features_from_file(self, file_path):
        """
        Extracts atomic and global features for a single material.
        """
        lattice, atom_data = self.parse_xyz(file_path)

        # 1. Atomic Features
        coords = np.array([[a[1], a[2], a[3]] for a in atom_data])
        types = [a[0] for a in atom_data]
        n_atoms = len(coords)

        # Centering
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        # Distance calculations
        # dists matrix shape: (N, 27*N)
        dists_matrix = self.compute_pbc_distances(coords, lattice)

        atomic_feats = []
        for i in range(n_atoms):
            # Filter out self-distance (which is 0.0)
            # The atom itself is in the supercell (offset 0,0,0).
            # We want nearest neighbors.
            # Sort distances
            d_i = np.sort(dists_matrix[i])

            # Remove 0 (self) - usually the first one
            d_i = d_i[d_i > 1e-6]

            # Nearest Neighbor Distance
            nn_dist = d_i[0] if len(d_i) > 0 else 0.0

            # Potential Proxy: Sum(1/d) for K nearest
            k = min(len(d_i), Config.POTENTIAL_K)
            if k > 0:
                potential = np.sum(1.0 / d_i[:k])
            else:
                potential = 0.0

            # One-hot encoding
            type_idx = Config.ATOM_TO_IDX[types[i]]
            one_hot = np.zeros(Config.NUM_ATOM_TYPES)
            one_hot[type_idx] = 1.0

            # Combine: OneHot(4) + Coords(3) + NN(1) + Pot(1)
            feat_vec = np.concatenate(
                [one_hot, centered_coords[i], [nn_dist], [potential]]
            )
            atomic_feats.append(feat_vec)

        atomic_feats = np.array(atomic_feats)  # (N, 9)

        # 2. Global Features
        # Lattice lengths
        lat_len = np.linalg.norm(lattice, axis=1)

        # Lattice angles
        # alpha: angle between v2 and v3
        # beta: angle between v1 and v3
        # gamma: angle between v1 and v2
        def angle(v1, v2):
            return np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )

        alpha = angle(lattice[1], lattice[2])
        beta = angle(lattice[0], lattice[2])
        gamma = angle(lattice[0], lattice[1])

        # Volume
        volume = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))

        # Density
        density = n_atoms / volume

        # Composition
        counts = {t: 0 for t in Config.ATOM_TYPES}
        for t in types:
            counts[t] += 1
        comp = (
            np.array([counts[t] for t in ["Al", "Ga", "In"]]) / n_atoms
        )  # Only cations as per metadata logic, but let's stick to metadata if possible.
        # Actually, let's recalculate composition to be safe and consistent with geometry

        global_vec = np.concatenate(
            [lat_len, [alpha, beta, gamma], [volume, density], comp]
        )  # 3 + 3 + 1 + 1 + 3 = 11

        return atomic_feats, global_vec

    def process_dataset(self, df, cache_path, load_cached=True, fit_scaler=False):
        """
        Process a dataframe (train/val/test) with caching.
        """
        # 1. Try Loading Cache
        if load_cached and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            try:
                data = np.load(
                    cache_path, allow_pickle=False
                )  # Pickle prohibited, use structured arrays logic

                # Reconstruct list of atomic features from flat array and counts
                flat_atomic = data["atomic_features_flat"]
                counts = data["atomic_counts"]
                global_features = data["global_features"]
                ids = data["ids"]

                # Rebuild list
                atomic_features_list = []
                cursor = 0
                for c in counts:
                    atomic_features_list.append(flat_atomic[cursor : cursor + c])
                    cursor += c

                targets = None
                if "targets" in data:
                    targets = data["targets"]

                # If we need to fit scaler (e.g. training set loaded from cache but this is a fresh run)
                # We must re-fit scalers on the loaded data if this is the training set.
                if fit_scaler:
                    self._fit_scalers(flat_atomic, global_features)

                # Apply scaling
                atomic_features_list, global_features = self._apply_scaling(
                    atomic_features_list, global_features
                )

                return MaterialDataset(
                    atomic_features_list, global_features, targets, ids
                )

            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        logger.info(f"Computing features for {len(df)} samples...")

        all_atomic_feats = []
        all_global_feats = []
        all_targets = []
        all_ids = []
        atomic_counts = []

        # Limit for debugging
        if Config.DEBUG_SAMPLE_SIZE:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        for idx, row in df.iterrows():
            try:
                a_feats, g_feats = self.extract_features_from_file(row["file_path"])

                all_atomic_feats.append(a_feats)
                all_global_feats.append(g_feats)
                atomic_counts.append(len(a_feats))
                all_ids.append(row["id"])

                if "formation_energy_ev_natom" in row:
                    # Log transform targets: log(1 + y)
                    # Note: formation energy can be 0, bandgap > 0.
                    t1 = np.log1p(row["formation_energy_ev_natom"])
                    t2 = np.log1p(row["bandgap_energy_ev"])
                    all_targets.append([t1, t2])
            except Exception as e:
                logger.error(f"Error processing {row['file_path']}: {e}")
                continue

        # Convert to arrays for caching
        flat_atomic = np.vstack(all_atomic_feats)
        global_features = np.vstack(all_global_feats)
        ids = np.array(all_ids)
        counts = np.array(atomic_counts)

        save_dict = {
            "atomic_features_flat": flat_atomic,
            "atomic_counts": counts,
            "global_features": global_features,
            "ids": ids,
        }

        targets = None
        if all_targets:
            targets = np.array(all_targets)
            save_dict["targets"] = targets

        # Save unscaled data to cache
        np.savez(cache_path, **save_dict)
        logger.info(f"Saved cache to {cache_path}")

        # 3. Fit Scalers (if train)
        if fit_scaler:
            self._fit_scalers(flat_atomic, global_features)

        # 4. Apply Scaling
        scaled_atomic_list, scaled_global = self._apply_scaling(
            all_atomic_feats, global_features
        )

        return MaterialDataset(scaled_atomic_list, scaled_global, targets, ids)

    def _fit_scalers(self, flat_atomic, global_features):
        """
        Fit scalers on training data.
        Atomic features: [OneHot(4), x, y, z, NN, Pot] -> Indices 4,5,6,7,8 are continuous.
        """
        # Scale only continuous atomic features (indices 4 to 8)
        self.atomic_scaler.fit(flat_atomic[:, 4:])
        self.global_scaler.fit(global_features)
        self.is_fitted = True
        logger.info("Scalers fitted.")

    def _apply_scaling(self, atomic_list, global_features):
        """
        Apply scaling to features.
        """
        if not self.is_fitted:
            raise RuntimeError("Scalers must be fitted before transforming data.")

        # Scale global
        scaled_global = self.global_scaler.transform(global_features)

        # Scale atomic
        scaled_atomic_list = []
        for af in atomic_list:
            # af shape (N, 9)
            # Copy categorical
            cat = af[:, :4]
            # Scale continuous
            cont = self.atomic_scaler.transform(af[:, 4:])
            # Recombine
            scaled_af = np.hstack([cat, cont])
            scaled_atomic_list.append(scaled_af)

        return scaled_atomic_list, scaled_global

    def get_datasets(self):
        """
        Main method to get all datasets.
        """
        # Load CSVs
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        logger.info("Processing Training Data...")
        train_dataset = self.process_dataset(
            train_df, Config.CACHE_PATH_TRAIN, load_cached=True, fit_scaler=True
        )

        logger.info("Processing Validation Data...")
        val_dataset = self.process_dataset(
            val_df, Config.CACHE_PATH_VAL, load_cached=True, fit_scaler=False
        )

        logger.info("Processing Test Data...")
        test_dataset = self.process_dataset(
            test_df, Config.CACHE_PATH_TEST, load_cached=True, fit_scaler=False
        )

        return train_dataset, val_dataset, test_dataset
