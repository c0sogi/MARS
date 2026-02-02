import os
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from library.config import Config


class FeatureExtractor:
    """
    Implements core physics-aware feature extraction logic for AMSP-DS strategy.
    Handles parsing, neighbor computation, and feature generation for both atomic and global streams.
    """

    def __init__(self):
        self.atom_map = Config.ATOM_MAP
        self.atomic_mass = Config.ATOMIC_MASS
        self.covalent_radius = Config.COVALENT_RADIUS
        self.electronegativity = Config.ELECTRONEGATIVITY

    def parse_xyz(self, file_path):
        """Reads geometry file using ASE."""
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        return read(full_path, format="aims")

    def _get_atomic_props(self, symbol):
        """Returns [mass, radius, electronegativity] for a given symbol."""
        return [
            self.atomic_mass.get(symbol, 0.0),
            self.covalent_radius.get(symbol, 0.0),
            self.electronegativity.get(symbol, 0.0),
        ]

    def compute_atomic_features(self, atoms):
        """
        Computes distortion-aware point process features for each atom.

        Features (17 dims):
        0-3: One-hot encoding (Al, Ga, In, O)
        4-6: Centered Cartesian coordinates (x, y, z)
        7:   Distance to nearest neighbor (d_min)
        8:   Local Packing Ratio (d_min / d_mean_12)
        9-12: Chemical Context (K=6)
        13-16: Chemical Context (K=24)

        Returns: numpy array of shape (N_atoms, 17)
        """
        n_atoms = len(atoms)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # Centering
        center_of_mass = atoms.get_center_of_mass()
        centered_pos = positions - center_of_mass

        # One-hot encoding
        one_hot = np.zeros((n_atoms, 4))
        for idx, s in enumerate(symbols):
            if s in self.atom_map:
                one_hot[idx, self.atom_map[s]] = 1.0

        # Neighbor search with PBC
        # Cutoff 10.0 A is sufficient to find >24 neighbors for dense crystals
        i_indices, j_indices, dists = neighbor_list("ijd", atoms, 10.0)

        # Prepare arrays for features
        d_min = np.zeros(n_atoms)
        packing_ratio = np.zeros(n_atoms)
        context_k6 = np.zeros((n_atoms, 4))
        context_k24 = np.zeros((n_atoms, 4))

        for a_idx in range(n_atoms):
            # Filter neighbors for atom a_idx
            mask = i_indices == a_idx

            a_dists = dists[mask]
            a_js = j_indices[mask]

            # Filter self-interactions (distance ~ 0)
            valid_mask = a_dists > 0.01
            a_dists = a_dists[valid_mask]
            a_js = a_js[valid_mask]

            # Sort by distance
            sorted_args = np.argsort(a_dists)
            sorted_dists = a_dists[sorted_args]
            sorted_js = a_js[sorted_args]

            # 1. Nearest Neighbor Distance (d_min)
            if len(sorted_dists) > 0:
                d_min[a_idx] = sorted_dists[0]
            else:
                d_min[a_idx] = 0.0

            # 2. Packing Ratio (d_min / d_mean_12)
            k_pack = Config.K_PACKING
            if len(sorted_dists) >= k_pack:
                d_mean_12 = np.mean(sorted_dists[:k_pack])
                packing_ratio[a_idx] = d_min[a_idx] / (d_mean_12 + 1e-8)
            elif len(sorted_dists) > 0:
                d_mean_12 = np.mean(sorted_dists)
                packing_ratio[a_idx] = d_min[a_idx] / (d_mean_12 + 1e-8)

            # 3. Chemical Contexts (Inverse Distance Weighted)
            def get_context(k_val):
                ctx = np.zeros(4)
                if len(sorted_dists) == 0:
                    return ctx

                # Take top K neighbors
                limit = min(len(sorted_dists), k_val)
                k_dists = sorted_dists[:limit]
                k_js = sorted_js[:limit]

                # Inverse distance weights
                weights = 1.0 / (k_dists + 1e-6)
                total_weight = np.sum(weights)

                for n_idx, w in zip(range(limit), weights):
                    # neighbor atom index in unit cell
                    neighbor_global_idx = k_js[n_idx]
                    # neighbor symbol
                    n_sym = symbols[neighbor_global_idx]
                    if n_sym in self.atom_map:
                        ctx[self.atom_map[n_sym]] += w

                if total_weight > 0:
                    ctx /= total_weight
                return ctx

            context_k6[a_idx] = get_context(Config.K_SHORT)
            context_k24[a_idx] = get_context(Config.K_MEDIUM)

        # Concatenate all atomic features
        features = np.hstack(
            [
                one_hot,  # 0-3
                centered_pos,  # 4-6
                d_min.reshape(-1, 1),  # 7
                packing_ratio.reshape(-1, 1),  # 8
                context_k6,  # 9-12
                context_k24,  # 13-16
            ]
        )

        return features

    def compute_global_features(self, row, atoms):
        """
        Computes anisotropic physics context features.

        Features (19 dims):
        0-5: Lattice parameters (a, b, c, alpha, beta, gamma)
        6-8: Aspect Ratios (a/b, b/c, c/a)
        9-12: Stoichiometry (Al, Ga, In, O fractions)
        13-15: Weighted Physics (Mass, Radius, Electronegativity)
        16: Cell Volume
        17: Atomic Density
        18: Total Atoms
        """
        # Lattice parameters
        a = row["lattice_vector_1_ang"]
        b = row["lattice_vector_2_ang"]
        c = row["lattice_vector_3_ang"]
        alpha = row["lattice_angle_alpha_degree"]
        beta = row["lattice_angle_beta_degree"]
        gamma = row["lattice_angle_gamma_degree"]

        # Aspect ratios
        r1 = a / (b + 1e-6)
        r2 = b / (c + 1e-6)
        r3 = c / (a + 1e-6)

        # Stoichiometry from atoms object to ensure consistency
        symbols = atoms.get_chemical_symbols()
        n_total = len(symbols)
        counts = {s: 0 for s in self.atom_map.keys()}
        for s in symbols:
            if s in counts:
                counts[s] += 1

        # Fractions
        fracs = [counts[s] / n_total for s in ["Al", "Ga", "In", "O"]]

        # Weighted Physics Properties
        w_mass = 0.0
        w_rad = 0.0
        w_en = 0.0

        for s, count in counts.items():
            frac = count / n_total
            props = self._get_atomic_props(s)
            w_mass += frac * props[0]
            w_rad += frac * props[1]
            w_en += frac * props[2]

        # Volume and Density
        # Volume formula for triclinic cell
        ca = np.cos(np.radians(alpha))
        cb = np.cos(np.radians(beta))
        cg = np.cos(np.radians(gamma))
        term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
        vol = a * b * c * np.sqrt(max(0, term))
        density = n_total / (vol + 1e-6)

        feats = np.array(
            [
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
                r1,
                r2,
                r3,
                fracs[0],
                fracs[1],
                fracs[2],
                fracs[3],
                w_mass,
                w_rad,
                w_en,
                vol,
                density,
                n_total,
            ]
        )

        return feats

    def process_dataset(self, df, load_cached_data=True, cache_name="train"):
        """
        Main processing function.
        Orchestrates feature extraction and caching.
        Returns dictionary containing raw numpy arrays.
        """
        cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return {
                "atomic_features": data["atomic_features"],
                "global_features": data["global_features"],
                "batch_indices": data["batch_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }

        print(f"Processing {len(df)} samples for {cache_name}...")

        all_atomic_feats = []
        all_global_feats = []
        all_batch_indices = []
        all_targets = []
        all_ids = []

        # Check if targets exist
        has_targets = "formation_energy_ev_natom" in df.columns

        for idx, row in df.iterrows():
            # Parse Geometry
            try:
                atoms = self.parse_xyz(row["file_path"])
            except Exception as e:
                print(f"Error parsing {row['file_path']}: {e}")
                continue

            # Atomic Features
            af = self.compute_atomic_features(atoms)
            all_atomic_feats.append(af)

            # Batch Indices (map atoms to crystal index)
            crystal_idx = len(all_ids)
            batch_idx = np.full(len(af), crystal_idx)
            all_batch_indices.append(batch_idx)

            # Global Features
            gf = self.compute_global_features(row, atoms)
            all_global_feats.append(gf)

            # Targets
            if has_targets:
                t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                all_targets.append(t)
            else:
                all_targets.append([0.0, 0.0])

            all_ids.append(row["id"])

        # Concatenate
        atomic_features = np.vstack(all_atomic_feats).astype(np.float32)
        global_features = np.vstack(all_global_feats).astype(np.float32)
        batch_indices = np.concatenate(all_batch_indices).astype(np.int64)
        targets = np.array(all_targets).astype(np.float32)
        ids = np.array(all_ids).astype(np.int64)

        # Save cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.savez_compressed(
            cache_path,
            atomic_features=atomic_features,
            global_features=global_features,
            batch_indices=batch_indices,
            targets=targets,
            ids=ids,
        )

        return {
            "atomic_features": atomic_features,
            "global_features": global_features,
            "batch_indices": batch_indices,
            "targets": targets,
            "ids": ids,
        }


class SelectiveScaler:
    """
    Scales features but skips one-hot encoded columns.
    Assumes one-hot columns are at the beginning of atomic features (indices 0-3).
    """

    def __init__(self, one_hot_dim=4):
        self.one_hot_dim = one_hot_dim
        self.atomic_mean = None
        self.atomic_std = None
        self.global_mean = None
        self.global_std = None

    def fit(self, atomic_feats, global_feats):
        # Atomic: Skip first 4 cols
        atomic_cont = atomic_feats[:, self.one_hot_dim :]
        self.atomic_mean = np.mean(atomic_cont, axis=0)
        self.atomic_std = np.std(atomic_cont, axis=0)
        # Avoid div by zero
        self.atomic_std[self.atomic_std < 1e-8] = 1.0

        # Global: Scale all
        self.global_mean = np.mean(global_feats, axis=0)
        self.global_std = np.std(global_feats, axis=0)
        self.global_std[self.global_std < 1e-8] = 1.0

    def transform(self, atomic_feats, global_feats):
        # Atomic
        atomic_scaled = atomic_feats.copy()
        atomic_cont = atomic_feats[:, self.one_hot_dim :]
        atomic_cont_scaled = (atomic_cont - self.atomic_mean) / self.atomic_std
        atomic_scaled[:, self.one_hot_dim :] = atomic_cont_scaled

        # Global
        global_scaled = (global_feats - self.global_mean) / self.global_std

        return atomic_scaled, global_scaled

    def save(self, path):
        np.savez(
            path,
            atomic_mean=self.atomic_mean,
            atomic_std=self.atomic_std,
            global_mean=self.global_mean,
            global_std=self.global_std,
            one_hot_dim=self.one_hot_dim,
        )

    def load(self, path):
        data = np.load(path)
        self.atomic_mean = data["atomic_mean"]
        self.atomic_std = data["atomic_std"]
        self.global_mean = data["global_mean"]
        self.global_std = data["global_std"]
        self.one_hot_dim = int(data["one_hot_dim"])
