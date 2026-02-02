import os
import numpy as np
import pandas as pd
from ase.io import read
from library.config import Config


class FeatureExtractor:
    """
    Extracts atomic and global features from material structures for the CEA-MS-DS model.
    Implements PBC-aware neighbor search, chemical context aggregation, and physics-based
    feature engineering.
    """

    def __init__(self):
        self.atom_map = Config.ATOM_MAP
        # Create a lookup table for atomic properties: [Mass, Radius, Electronegativity]
        # Shape: (4, 3) corresponding to Al, Ga, In, O
        self.prop_lookup = np.zeros((4, 3))
        for sym, idx in self.atom_map.items():
            if sym in Config.ATOMIC_PROPS:
                self.prop_lookup[idx] = Config.ATOMIC_PROPS[sym]

    def _get_pbc_neighbors(self, atoms, k_max):
        """
        Computes k-nearest neighbors under Periodic Boundary Conditions (PBC).

        Args:
            atoms: ASE Atoms object.
            k_max: Number of neighbors to find.

        Returns:
            distances: (N_atoms, k_max) array of distances.
            indices: (N_atoms, k_max) array of neighbor atom indices.
        """
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        n_atoms = len(atoms)

        # Generate image offsets for 3x3x3 supercell to ensure enough neighbors
        # even for small unit cells.
        ranges = [-1, 0, 1]
        shifts = []
        for i in ranges:
            for j in ranges:
                for k in ranges:
                    shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
        shifts = np.array(shifts)  # (27, 3)

        # Broadcast positions to find distances to all atoms in all 27 images
        # pos: (N, 3)
        # all_image_pos: (27*N, 3)
        all_image_pos = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

        # Map supercell indices back to unit cell indices (0 to N-1)
        all_indices = np.tile(np.arange(n_atoms), 27)

        neighbor_dists = []
        neighbor_indices = []

        # Compute distances for each atom
        for i in range(n_atoms):
            # Distance from atom i to all atoms in supercell
            d = np.linalg.norm(all_image_pos - pos[i], axis=1)

            # Sort by distance
            sorted_args = np.argsort(d)

            # Filter out self-interaction (distance ~ 0)
            # We look for d > epsilon. The self-atom is at d=0.
            valid_mask = d[sorted_args] > 1e-4
            valid_args = sorted_args[valid_mask]

            # Take top k_max
            # If for some reason we don't have enough (unlikely with 3x3x3), take what we have
            k_actual = min(len(valid_args), k_max)

            current_dists = d[valid_args[:k_actual]]
            current_indices = all_indices[valid_args[:k_actual]]

            # Pad if necessary (though 27 images usually suffice for k=24)
            if k_actual < k_max:
                pad_width = k_max - k_actual
                current_dists = np.pad(current_dists, (0, pad_width), "edge")
                current_indices = np.pad(current_indices, (0, pad_width), "edge")

            neighbor_dists.append(current_dists)
            neighbor_indices.append(current_indices)

        return np.array(neighbor_dists), np.array(neighbor_indices)

    def _get_atomic_features(self, atoms, neighbors_dists, neighbors_indices):
        """
        Constructs the 21-dimensional atomic feature vector.

        Features:
          0-3:   Atomic Identity (One-Hot)
          4-7:   Nearest Neighbor Identity (One-Hot)
          8-10:  Centered XYZ Coordinates
          11:    Nearest Neighbor Distance (d_min)
          12:    Local Packing Ratio (d_min / d_mean_12)
          13-16: Chemical Context K=6 (Weighted Composition)
          17-20: Chemical Context K=24 (Weighted Composition)
        """
        n_atoms = len(atoms)
        symbols = atoms.get_chemical_symbols()
        pos = atoms.get_positions()

        # Map symbols to integer indices
        type_indices = np.array([self.atom_map[s] for s in symbols])

        # 1. Atomic Identity (One-Hot) -> (N, 4)
        identity_onehot = np.eye(4)[type_indices]

        # 2. Nearest Neighbor Identity (One-Hot) -> (N, 4)
        # Neighbor 0 is the closest one
        nn_indices = neighbors_indices[:, 0]
        nn_type_indices = type_indices[nn_indices]
        nn_identity_onehot = np.eye(4)[nn_type_indices]

        # 3. Spatial Context (Centered Coords) -> (N, 3)
        centroid = np.mean(pos, axis=0)
        centered_pos = pos - centroid

        # 4. Nearest Neighbor Distance (d_min) -> (N, 1)
        d_min = neighbors_dists[:, 0:1]

        # 5. Local Packing Ratio -> (N, 1)
        # Ratio of d_min to mean distance of 12 nearest neighbors
        k_pack = 12
        if neighbors_dists.shape[1] >= k_pack:
            d_mean_12 = np.mean(neighbors_dists[:, :k_pack], axis=1, keepdims=True)
        else:
            d_mean_12 = np.mean(neighbors_dists, axis=1, keepdims=True)
        packing_ratio = d_min / (d_mean_12 + 1e-8)

        # 6 & 7. Multi-Scale Chemical Contexts -> (N, 4) each
        # Inverse-distance weighted average of neighbor identities
        def get_context_vector(k):
            k_eff = min(neighbors_dists.shape[1], k)

            d_k = neighbors_dists[:, :k_eff]  # (N, k)
            idx_k = neighbors_indices[:, :k_eff]  # (N, k)

            # Weights = 1 / d
            weights = 1.0 / (d_k + 1e-6)  # (N, k)

            # Get one-hot types for all neighbors
            # Flatten indices to map, then reshape
            flat_idx = idx_k.flatten()
            flat_types = type_indices[flat_idx]
            flat_onehot = np.eye(4)[flat_types]  # (N*k, 4)
            neigh_onehot = flat_onehot.reshape(n_atoms, k_eff, 4)

            # Weighted sum: sum_k ( weight_ik * onehot_ik )
            # (N, k, 1) * (N, k, 4) -> (N, k, 4) -> sum -> (N, 4)
            weighted_sum = np.sum(weights[:, :, None] * neigh_onehot, axis=1)

            # Normalize by sum of weights
            sum_weights = np.sum(weights, axis=1, keepdims=True)
            context = weighted_sum / (sum_weights + 1e-8)
            return context

        context_6 = get_context_vector(6)
        context_24 = get_context_vector(24)

        # Concatenate all features
        features = np.concatenate(
            [
                identity_onehot,  # 0-3
                nn_identity_onehot,  # 4-7
                centered_pos,  # 8-10
                d_min,  # 11
                packing_ratio,  # 12
                context_6,  # 13-16
                context_24,  # 17-20
            ],
            axis=1,
        )

        return features

    def _get_global_features(self, atoms):
        """
        Constructs the 21-dimensional global feature vector.

        Features:
          0-2:   Lattice Vector Lengths
          3-5:   Lattice Angles
          6:     Unit Cell Volume
          7-9:   Lattice Aspect Ratios
          10:    Atomic Density
          11:    Total Number of Atoms
          12-14: Stoichiometry (Al, Ga, In fractions)
          15-17: Mean Physical Properties (Mass, Radius, EN)
          18-20: Std Dev Physical Properties
        """
        # Geometric Features
        lengths = atoms.cell.lengths()  # (3,)
        angles = atoms.cell.angles()  # (3,)
        volume = atoms.get_volume()  # Scalar

        a, b, c = lengths
        aspect_ratios = np.array([a / b, b / c, c / a])  # (3,)

        # Structural Features
        n_atoms = len(atoms)
        density = n_atoms / volume

        # Chemical Features
        symbols = atoms.get_chemical_symbols()
        type_indices = [self.atom_map[s] for s in symbols]

        # Stoichiometry
        counts = np.zeros(4)
        for idx in type_indices:
            counts[idx] += 1
        fracs = counts / n_atoms
        stoich_3 = fracs[:3]  # Al, Ga, In fractions (O is implicit)

        # Physical Property Statistics
        # Get properties for every atom in the cell
        atom_props = self.prop_lookup[type_indices]  # (N, 3)

        mean_props = np.mean(atom_props, axis=0)  # (3,)
        std_props = np.std(atom_props, axis=0)  # (3,)

        # Concatenate
        global_feat = np.concatenate(
            [
                lengths,  # 0-2
                angles,  # 3-5
                [volume],  # 6
                aspect_ratios,  # 7-9
                [density],  # 10
                [n_atoms],  # 11
                stoich_3,  # 12-14
                mean_props,  # 15-17
                std_props,  # 18-20
            ]
        )

        return global_feat

    def process_data(self, df, load_cached_data=True, split_name="train"):
        """
        Processes a dataframe of materials into feature arrays.

        Args:
            df: DataFrame containing 'file_path' and optionally targets.
            load_cached_data: If True, attempts to load from cache first.
            split_name: Name of the split (train/val/test) for caching.

        Returns:
            Dictionary containing:
                X_atomic: (Total_Atoms, 21)
                X_global: (N_Samples, 21)
                y: (N_Samples, 2)
                batch_idx: (Total_Atoms,) mapping atoms to samples
        """
        cache_file = os.path.join(Config.WORKING_DIR, f"{split_name}_data.npz")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {split_name} data from {cache_file}...")
            try:
                data = np.load(cache_file)
                return {
                    "X_atomic": data["X_atomic"],
                    "X_global": data["X_global"],
                    "y": data["y"],
                    "batch_idx": data["batch_idx"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Processing {split_name} data from scratch...")

        all_atomic_feats = []
        all_global_feats = []
        all_targets = []
        batch_indices = []

        for i, row in df.iterrows():
            # Load geometry
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            atoms = read(file_path)

            # Compute Neighbors
            dists, indices = self._get_pbc_neighbors(atoms, Config.K_NEIGHBORS_MAX)

            # Extract Features
            atomic_f = self._get_atomic_features(atoms, dists, indices)
            global_f = self._get_global_features(atoms)

            all_atomic_feats.append(atomic_f)
            all_global_feats.append(global_f)

            # Handle Targets
            if "formation_energy_ev_natom" in row:
                t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                all_targets.append(t)
            else:
                all_targets.append([0.0, 0.0])  # Placeholder for test

            # Create batch index for this sample
            # i is the loop index, but we want contiguous 0..N-1 indices for the batch
            sample_idx = len(all_global_feats) - 1
            n_atoms_in_sample = len(atoms)
            batch_indices.append(np.full(n_atoms_in_sample, sample_idx))

        # Concatenate all lists into arrays
        X_atomic = np.vstack(all_atomic_feats).astype(np.float32)
        X_global = np.vstack(all_global_feats).astype(np.float32)
        y = np.array(all_targets).astype(np.float32)
        batch_idx = np.concatenate(batch_indices).astype(np.int64)

        # Save to cache
        np.savez(
            cache_file, X_atomic=X_atomic, X_global=X_global, y=y, batch_idx=batch_idx
        )
        print(f"Saved {split_name} data to {cache_file}")

        return {
            "X_atomic": X_atomic,
            "X_global": X_global,
            "y": y,
            "batch_idx": batch_idx,
        }


def get_data_loaders(debug_size=None):
    """
    Helper function to load train, val, and test data.

    Args:
        debug_size: If integer, limits the number of samples for debugging.

    Returns:
        train_data, val_data, test_data (dictionaries)
    """
    extractor = FeatureExtractor()

    # Read metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Subsample if debugging
    if debug_size is not None:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]
        print(f"Debug mode enabled: Processing {debug_size} samples per split.")
        suffix = f"_debug_{debug_size}"
    else:
        suffix = ""

    # Process each split
    train_data = extractor.process_data(
        train_df, load_cached_data=True, split_name=f"train{suffix}"
    )
    val_data = extractor.process_data(
        val_df, load_cached_data=True, split_name=f"val{suffix}"
    )
    test_data = extractor.process_data(
        test_df, load_cached_data=True, split_name=f"test{suffix}"
    )

    return train_data, val_data, test_data
