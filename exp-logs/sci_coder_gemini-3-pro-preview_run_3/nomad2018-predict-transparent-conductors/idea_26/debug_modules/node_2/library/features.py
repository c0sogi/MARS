import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from scipy.special import sph_harm
from scipy.stats import entropy
import warnings

# Import configuration and data loader
from library.config import Config
from library.data_loader import load_metadata, read_geometry

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class GeometricFeaturizer:
    """
    Implements Distributional Multi-Order Geometric Fingerprinting (DMGF).
    """

    def __init__(self):
        self.elements = Config.ELEMENTS
        self.rdf_cutoff = Config.RDF_CUTOFF
        self.rdf_bins = Config.RDF_BINS
        self.angle_cutoff = Config.ANGLE_NEIGHBOR_CUTOFF
        self.local_cutoff = Config.LOCAL_ENV_CUTOFF
        self.steinhardt_ls = Config.STEINHARDT_L
        self.percentiles = Config.PERCENTILES

        # Pre-compute pairs for RDF
        self.pairs = []
        for i, e1 in enumerate(self.elements):
            for e2 in self.elements[i:]:
                self.pairs.append(tuple(sorted((e1, e2))))

    def _compute_global(self, atoms):
        """Computes global descriptors: Volume and Density."""
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 1e-6 else 0.0
        return np.array([vol, density])

    def _compute_radial(self, atoms, i_indices, j_indices, dists, symbols):
        """
        Computes element-resolved RDFs and bond length percentiles.
        Uses pre-calculated neighbor list arrays.
        """
        features = []

        # Total number of atoms to normalize RDF
        n_atoms = len(atoms)

        for e1, e2 in self.pairs:
            # Mask for this pair
            # i_indices are indices of central atoms, j_indices are neighbors
            # We need to check the chemical symbols of i and j

            # Vectorized check
            sym_i = symbols[i_indices]
            sym_j = symbols[j_indices]

            # Check match (undirected)
            mask = ((sym_i == e1) & (sym_j == e2)) | ((sym_i == e2) & (sym_j == e1))
            pair_dists = dists[mask]

            # 1. RDF Histogram
            hist, _ = np.histogram(
                pair_dists, bins=self.rdf_bins, range=(0, self.rdf_cutoff)
            )
            # Normalize by number of atoms (density independent intensity)
            hist = hist.astype(float) / n_atoms
            features.extend(hist)

            # 2. Bond Length Percentiles
            if len(pair_dists) > 0:
                pcts = np.percentile(pair_dists, self.percentiles)
            else:
                pcts = np.zeros(len(self.percentiles))
            features.extend(pcts)

        return np.array(features)

    def _compute_angular(self, atoms, i_indices, j_indices, vectors, symbols):
        """
        Computes bond angle percentiles for O-Metal-O and Metal-O-Metal.
        """
        # Filter neighbors within angle cutoff
        # Note: The passed indices/vectors are for RDF cutoff (6.0), we need to filter for 3.0
        mask_cutoff = np.linalg.norm(vectors, axis=1) <= self.angle_cutoff

        idx_i = i_indices[mask_cutoff]
        idx_j = j_indices[mask_cutoff]
        vecs = vectors[mask_cutoff]
        syms_i = symbols[idx_i]
        syms_j = symbols[idx_j]

        # Normalize vectors for angle calculation
        norms = np.linalg.norm(vecs, axis=1)
        # Avoid division by zero
        norms[norms < 1e-6] = 1.0
        vecs_norm = vecs / norms[:, np.newaxis]

        # We need to construct adjacency list to iterate efficiently
        # Group neighbors by central atom index
        adj = {}
        for k, center_idx in enumerate(idx_i):
            if center_idx not in adj:
                adj[center_idx] = []
            adj[center_idx].append(
                (k, idx_j[k], syms_j[k])
            )  # store index in vecs_norm, neighbor idx, neighbor symbol

        # Lists to store angles
        # O-Metal-O: Center is Metal, Neighbors are O
        # Metal-O-Metal: Center is O, Neighbors are Metal
        angles_omo = []
        angles_mom = []

        metals = set(["Al", "Ga", "In"])

        for center_idx, neighbors in adj.items():
            center_sym = symbols[center_idx]

            # If center is Metal
            if center_sym in metals:
                # Find Oxygen neighbors
                o_neighbors = [v for v in neighbors if v[2] == "O"]
                # Compute angles between all pairs of O neighbors
                n_o = len(o_neighbors)
                if n_o >= 2:
                    for a in range(n_o):
                        for b in range(a + 1, n_o):
                            # Dot product of normalized vectors
                            vec_a = vecs_norm[o_neighbors[a][0]]
                            vec_b = vecs_norm[o_neighbors[b][0]]
                            cosine = np.dot(vec_a, vec_b)
                            # Clip for numerical stability
                            cosine = np.clip(cosine, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cosine))
                            angles_omo.append(angle)

            # If center is Oxygen
            elif center_sym == "O":
                # Find Metal neighbors
                m_neighbors = [v for v in neighbors if v[2] in metals]
                n_m = len(m_neighbors)
                if n_m >= 2:
                    for a in range(n_m):
                        for b in range(a + 1, n_m):
                            vec_a = vecs_norm[m_neighbors[a][0]]
                            vec_b = vecs_norm[m_neighbors[b][0]]
                            cosine = np.dot(vec_a, vec_b)
                            cosine = np.clip(cosine, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cosine))
                            angles_mom.append(angle)

        # Compute percentiles
        feats = []
        for dist_list in [angles_omo, angles_mom]:
            if len(dist_list) > 0:
                feats.extend(np.percentile(dist_list, self.percentiles))
            else:
                feats.extend(np.zeros(len(self.percentiles)))

        return np.array(feats)

    def _compute_local_order(self, atoms, i_indices, j_indices, vectors, symbols):
        """
        Computes ECoN and Steinhardt Order Parameters (Q4, Q6), aggregated by element.
        """
        # Filter for local cutoff
        mask_cutoff = np.linalg.norm(vectors, axis=1) <= self.local_cutoff

        idx_i = i_indices[mask_cutoff]
        vecs = vectors[mask_cutoff]

        # Prepare data structures
        n_atoms = len(atoms)
        unique_elements = self.elements  # Al, Ga, In, O

        # Per-atom metrics
        # Coordination Number (count of neighbors within cutoff)
        coord_nums = np.zeros(n_atoms)
        # Q_l parameters
        q_l_vals = {l: np.zeros(n_atoms) for l in self.steinhardt_ls}

        # Group vectors by central atom
        # We can use bincount for coordination number
        counts = np.bincount(idx_i, minlength=n_atoms)
        coord_nums[:] = counts

        # For Steinhardt, we need to iterate because of spherical harmonics summation
        # Use an adjacency map for vectors
        adj_vecs = {}
        for k, center_idx in enumerate(idx_i):
            if center_idx not in adj_vecs:
                adj_vecs[center_idx] = []
            adj_vecs[center_idx].append(vecs[k])

        for i in range(n_atoms):
            if i in adj_vecs:
                vs = np.array(adj_vecs[i])
                # Convert to spherical coordinates
                # x, y, z -> r, theta, phi
                # theta = arccos(z/r), phi = arctan2(y, x)
                rs = np.linalg.norm(vs, axis=1)
                # Avoid div by zero
                rs[rs < 1e-6] = 1.0

                thetas = np.arccos(vs[:, 2] / rs)
                phis = np.arctan2(vs[:, 1], vs[:, 0])

                for l in self.steinhardt_ls:
                    # Sum Y_lm over neighbors
                    # m goes from -l to l
                    # scipy sph_harm(m, l, phi, theta)

                    sum_sq_mag = 0.0
                    for m in range(-l, l + 1):
                        y_lm = sph_harm(m, l, phis, thetas)
                        # Average over neighbors
                        q_lm = np.mean(y_lm)
                        sum_sq_mag += np.abs(q_lm) ** 2

                    q_l = np.sqrt(4 * np.pi / (2 * l + 1) * sum_sq_mag)
                    q_l_vals[l][i] = q_l
            else:
                # No neighbors
                pass

        # Aggregate by element type
        features = []
        atom_symbols = np.array(atoms.get_chemical_symbols())

        for elem in unique_elements:
            mask = atom_symbols == elem
            if np.any(mask):
                # Coordination Number Distribution
                cn_dist = coord_nums[mask]
                features.extend(np.percentile(cn_dist, self.percentiles))

                # Q_l Distributions
                for l in self.steinhardt_ls:
                    ql_dist = q_l_vals[l][mask]
                    features.extend(np.percentile(ql_dist, self.percentiles))
            else:
                # Element not present in structure
                # Fill with zeros
                features.extend(np.zeros(len(self.percentiles)))  # CN
                for _ in self.steinhardt_ls:
                    features.extend(np.zeros(len(self.percentiles)))  # Ql

        return np.array(features)

    def featurize(self, atoms):
        """
        Main method to convert ASE Atoms to feature vector.
        """
        # 1. Global
        global_feats = self._compute_global(atoms)

        # 2. Neighbor List (computed once for max cutoff)
        # RDF_CUTOFF is usually the largest (6.0 A)
        # neighbor_list returns (i, j, D, d) for "ijDd"
        # i: index of central atom
        # j: index of neighbor atom
        # D: vector pointing from i to j
        # d: distance
        i_indices, j_indices, vectors, dists = neighbor_list(
            "ijDd", atoms, self.rdf_cutoff
        )
        symbols = np.array(atoms.get_chemical_symbols())

        # 3. Radial Features
        radial_feats = self._compute_radial(atoms, i_indices, j_indices, dists, symbols)

        # 4. Angular Features
        # Uses subsets of the neighbor list (cutoff 3.0)
        angular_feats = self._compute_angular(
            atoms, i_indices, j_indices, vectors, symbols
        )

        # 5. Local Order Features
        # Uses subsets of the neighbor list (cutoff 3.5)
        local_feats = self._compute_local_order(
            atoms, i_indices, j_indices, vectors, symbols
        )

        # Concatenate
        return np.concatenate([global_feats, radial_feats, angular_feats, local_feats])


def process_data(split: str, load_cached_data: bool = True):
    """
    Loads metadata, computes features (or loads from cache), and returns a DataFrame.
    """
    # Determine Cache Path
    if split == "train":
        cache_path = Config.TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = Config.VAL_FEATURES_PATH
    elif split == "test":
        cache_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError("Invalid split")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {split} from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Computing features for {split}...")
    df_meta = load_metadata(split)

    featurizer = GeometricFeaturizer()
    feature_list = []
    ids = []

    # Loop through metadata
    # Note: No progress bar as per instructions
    for idx, row in df_meta.iterrows():
        try:
            atoms = read_geometry(row["file_path"])
            feats = featurizer.featurize(atoms)
            feature_list.append(feats)
            ids.append(row["id"])
        except Exception as e:
            # Handle potential read errors
            print(f"Error processing ID {row['id']}: {e}")
            pass

    if not feature_list:
        raise RuntimeError("No features computed.")

    # Create DataFrame
    # Column names are auto-generated
    cols = [f"feat_{i}" for i in range(len(feature_list[0]))]
    df_features = pd.DataFrame(feature_list, columns=cols)
    df_features["id"] = ids

    # Merge with targets if available
    if split in ["train", "val"]:
        # Ensure alignment on ID
        df_features = df_features.merge(
            df_meta[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]],
            on="id",
            how="left",
        )

    # Save to cache
    print(f"Saving features to {cache_path}...")
    df_features.to_parquet(cache_path, index=False)

    return df_features
