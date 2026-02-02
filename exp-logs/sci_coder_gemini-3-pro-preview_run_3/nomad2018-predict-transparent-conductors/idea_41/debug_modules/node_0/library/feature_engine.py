import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    ATOMIC_SPECIES,
    METALS,
    ANIONS,
    BVS_PARAMS,
    BVS_B,
    RDF_CUTOFF,
    RDF_BIN_WIDTH,
    BONDING_CUTOFF,
    RANDOM_SEED,
)


class FingerprintGenerator:
    """
    Generates Focused Chemo-Structural Distributional Fingerprints for oxide materials.
    """

    def __init__(self):
        self.rdf_bins = np.arange(0, RDF_CUTOFF + 1e-6, RDF_BIN_WIDTH)
        self.percentiles = [0, 25, 50, 75, 100]
        self.species = ATOMIC_SPECIES

    def generate(self, atoms):
        """
        Generates a feature vector for a single ASE Atoms object.
        """
        features = {}

        # 1. Macroscopic Features
        macro_feats = self._compute_macroscopic(atoms)
        features.update(macro_feats)

        # 2. Radial Distribution Functions (M-O pairs)
        rdf_feats = self._compute_metal_oxygen_rdf(atoms)
        features.update(rdf_feats)

        # 3. Local Site Metrics
        local_metrics = self._compute_local_site_metrics(atoms)
        # Aggregate local metrics
        agg_local = self._aggregate_distributions(local_metrics, prefix="local")
        features.update(agg_local)

        # 4. Topological Features (Angles)
        angles = self._compute_angle_distributions(atoms)
        # Aggregate angles
        angle_feats = self._aggregate_angle_distributions(angles)
        features.update(angle_feats)

        return features

    def _compute_macroscopic(self, atoms):
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 1e-6 else 0.0
        return {"vol_per_atom": vol / len(atoms), "density": density}

    def _compute_metal_oxygen_rdf(self, atoms):
        """
        Computes RDF for Metal-Oxygen pairs.
        """
        # i: indices of center atoms, j: indices of neighbors, d: distances
        i_indices, j_indices, d_values = neighbor_list("ijd", atoms, cutoff=RDF_CUTOFF)

        symbols = np.array(atoms.get_chemical_symbols())

        feats = {}

        # Iterate over Metal-Oxygen pairs
        for metal in METALS:
            pair_label = f"RDF_{metal}_O"

            # Mask for pairs where (atom_i is Metal AND atom_j is O)
            # Since neighbor_list returns both i-j and j-i, we can just look for i=Metal, j=O
            if len(i_indices) > 0:
                mask_i = symbols[i_indices] == metal
                mask_j = symbols[j_indices] == "O"
                mask = mask_i & mask_j
                dists = d_values[mask]
            else:
                dists = np.array([])

            # Histogram
            hist, _ = np.histogram(dists, bins=self.rdf_bins)

            # Normalize by total number of atoms (intensive property)
            hist = hist.astype(float) / len(atoms)

            for k, count in enumerate(hist):
                feats[f"{pair_label}_bin_{k}"] = count

        return feats

    def _compute_local_site_metrics(self, atoms):
        """
        Computes BVS, ECoN, and Local Anisotropy for each atom.
        """
        # Use bonding cutoff
        i_indices, j_indices, d_values, v_values = neighbor_list(
            "ijdD", atoms, cutoff=BONDING_CUTOFF
        )

        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        # Initialize arrays
        bvs_values = np.zeros(n_atoms)
        econ_values = np.zeros(n_atoms)
        anisotropy_values = np.zeros(n_atoms)

        if len(i_indices) > 0:
            # Create a DataFrame for easier grouping
            df_neighbors = pd.DataFrame(
                {
                    "i": i_indices,
                    "j": j_indices,
                    "d": d_values,
                    "vx": v_values[:, 0],
                    "vy": v_values[:, 1],
                    "vz": v_values[:, 2],
                }
            )

            # Add symbols
            df_neighbors["sym_i"] = [symbols[k] for k in df_neighbors["i"]]
            df_neighbors["sym_j"] = [symbols[k] for k in df_neighbors["j"]]

            # 1. BVS (Only Metal-Oxygen pairs contribute)
            # Formula: exp((R0 - d) / B)
            def get_bvs_term(row):
                pair = (row["sym_i"], row["sym_j"])
                if pair in BVS_PARAMS:
                    r0 = BVS_PARAMS[pair]
                    return np.exp((r0 - row["d"]) / BVS_B)
                return 0.0

            df_neighbors["bvs_term"] = df_neighbors.apply(get_bvs_term, axis=1)
            bvs_sums = df_neighbors.groupby("i")["bvs_term"].sum()
            bvs_values[bvs_sums.index] = bvs_sums.values

            # 2. ECoN (Effective Coordination Number)
            # Using a cosine-based smooth cutoff
            # f(r) = 0.5 * (1 + cos(pi * r / R_cut))
            df_neighbors["econ_term"] = 0.5 * (
                1.0 + np.cos(np.pi * df_neighbors["d"] / BONDING_CUTOFF)
            )
            econ_sums = df_neighbors.groupby("i")["econ_term"].sum()
            econ_values[econ_sums.index] = econ_sums.values

            # 3. Local Anisotropy
            # Magnitude of vector sum of bond directions.
            # Normalize vectors first
            norms = np.linalg.norm(df_neighbors[["vx", "vy", "vz"]].values, axis=1)
            # Avoid division by zero
            norms[norms == 0] = 1.0

            vecs = df_neighbors[["vx", "vy", "vz"]].values / norms[:, None]

            # Sum vectors for each i
            # Groupby sum on components
            vec_sums = (
                pd.DataFrame(vecs, index=df_neighbors.index)
                .groupby(df_neighbors["i"])
                .sum()
            )

            # Calculate magnitude
            aniso_mags = np.linalg.norm(vec_sums.values, axis=1)
            anisotropy_values[vec_sums.index] = aniso_mags

        # Return dict of arrays keyed by metric name
        return {
            "BVS": bvs_values,
            "ECoN": econ_values,
            "Anisotropy": anisotropy_values,
            "symbols": symbols,
        }

    def _compute_angle_distributions(self, atoms):
        """
        Computes M-O-M and O-M-O bond angles.
        """
        # We need connectivity. Re-use BONDING_CUTOFF.
        i_indices, j_indices, d_values, v_values = neighbor_list(
            "ijdD", atoms, cutoff=BONDING_CUTOFF
        )

        symbols = atoms.get_chemical_symbols()

        # Build adjacency: atom_idx -> list of (neighbor_idx, vector_i_to_j)
        adj = {k: [] for k in range(len(atoms))}
        for idx, i in enumerate(i_indices):
            j = j_indices[idx]
            v = v_values[idx]
            adj[i].append((j, v))

        omo_angles = []
        mom_angles = []

        for i, neighbors in adj.items():
            center_sym = symbols[i]

            # Filter neighbors based on chemistry
            relevant_neighbors = []
            target_list = None

            if center_sym in METALS:
                # Look for O neighbors -> O-M-O
                relevant_neighbors = [n for n in neighbors if symbols[n[0]] == "O"]
                target_list = omo_angles
            elif center_sym == "O":
                # Look for Metal neighbors -> M-O-M
                relevant_neighbors = [n for n in neighbors if symbols[n[0]] in METALS]
                target_list = mom_angles

            if target_list is None:
                continue

            # Compute angles between all pairs of relevant neighbors
            n_neigh = len(relevant_neighbors)
            if n_neigh < 2:
                continue

            for idx1 in range(n_neigh):
                for idx2 in range(idx1 + 1, n_neigh):
                    v1 = relevant_neighbors[idx1][1]
                    v2 = relevant_neighbors[idx2][1]

                    # Angle calculation
                    # cos_theta = (v1 . v2) / (|v1| |v2|)
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        dot = np.dot(v1, v2)
                        cos_theta = dot / (n1 * n2)
                        # Clip for numerical stability
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_theta))
                        target_list.append(angle)

        return {"OMO_Angles": omo_angles, "MOM_Angles": mom_angles}

    def _aggregate_distributions(self, metrics_dict, prefix=""):
        """
        Aggregates atom-wise metrics into percentiles grouped by species.
        """
        symbols = np.array(metrics_dict["symbols"])
        feats = {}

        metric_names = [k for k in metrics_dict.keys() if k != "symbols"]

        for species in self.species:
            mask = symbols == species
            count = np.sum(mask)

            for metric in metric_names:
                values = metrics_dict[metric][mask]

                if count > 0:
                    percs = np.percentile(values, self.percentiles)
                else:
                    percs = [np.nan] * len(self.percentiles)

                for p, val in zip(self.percentiles, percs):
                    feats[f"{prefix}_{species}_{metric}_p{p}"] = val

        return feats

    def _aggregate_angle_distributions(self, angles_dict):
        """
        Aggregates angle lists into percentiles.
        """
        feats = {}
        for key, angle_list in angles_dict.items():
            if len(angle_list) > 0:
                percs = np.percentile(angle_list, self.percentiles)
            else:
                percs = [np.nan] * len(self.percentiles)

            for p, val in zip(self.percentiles, percs):
                feats[f"topo_{key}_p{p}"] = val
        return feats


def process_dataset(metadata_df, load_cached_data=True):
    """
    Processes the dataset to generate features.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Use a hash or length of df to create a unique filename, or just a standard name
    # Appending length helps avoid using stale cache if dataset size changes
    cache_file = os.path.join(CACHE_DIR, f"features_{len(metadata_df)}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return pd.read_parquet(cache_file)

    print("Computing features from scratch...")
    generator = FingerprintGenerator()
    feature_list = []

    # Iterate over metadata
    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        mat_id = row["id"]

        try:
            atoms = ase.io.read(file_path)
            feats = generator.generate(atoms)
            feats["id"] = mat_id
            feature_list.append(feats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Append a dict with just ID to keep alignment, filled with NaNs
            feature_list.append({"id": mat_id})

    df_features = pd.DataFrame(feature_list)

    # Ensure 'id' columns are same type for merging
    df_features["id"] = df_features["id"].astype(int)
    metadata_df_copy = metadata_df.copy()
    metadata_df_copy["id"] = metadata_df_copy["id"].astype(int)

    # Merge with original metadata
    df_final = pd.merge(metadata_df_copy, df_features, on="id", how="left")

    # Save to cache
    df_final.to_parquet(cache_file)
    print(f"Saved features to {cache_file}")

    return df_final
