import os
import numpy as np
import pandas as pd
import ase.neighborlist
from ase.data import chemical_symbols, atomic_masses

from library.config import (
    WORKING_DIR,
    RDF_CUTOFF,
    RDF_NUM_BINS,
    RDF_SIGMA,
    CONNECTIVITY_CUTOFF,
    LOCAL_ENV_CUTOFF,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import (
    compute_percentiles,
    calculate_bvs,
    calculate_econ,
    calculate_local_anisotropy,
    get_pbc_displacement,
    OXIDATION_STATES,
)
from library.data_loader import load_metadata, load_structure


class StructureFeaturizer:
    def __init__(self):
        self.rdf_bins = np.linspace(0, RDF_CUTOFF, RDF_NUM_BINS + 1)
        self.r_vals = 0.5 * (self.rdf_bins[1:] + self.rdf_bins[:-1])
        # Elements of interest
        self.metals = ["Al", "Ga", "In"]
        self.anion = "O"
        self.all_elements = self.metals + [self.anion]

    def compute_macroscopic(self, atoms):
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        num_atoms = len(atoms)
        return {
            "vol_per_atom": vol / num_atoms if num_atoms > 0 else 0,
            "density": mass / vol if vol > 0 else 0,
        }

    def compute_rdf(self, atoms):
        """
        Computes element-resolved RDFs.
        """
        features = {}
        # Define pairs of interest: Metal-O and Metal-Metal
        pairs = []
        for m in self.metals:
            pairs.append((m, "O"))
            for m2 in self.metals:
                # Lexicographical order to avoid duplicates (Al-Ga vs Ga-Al)
                if m <= m2:
                    pairs.append((m, m2))

        # Get all distances
        # We use neighbor list to get distances efficiently with PBC
        # self_interaction=False to avoid dist=0
        i_indices, j_indices, distances = ase.neighborlist.neighbor_list(
            "ijd", atoms, RDF_CUTOFF, self_interaction=False
        )

        symbols = np.array(atoms.get_chemical_symbols())

        for e1, e2 in pairs:
            # Mask for specific pair
            mask = ((symbols[i_indices] == e1) & (symbols[j_indices] == e2)) | (
                (symbols[i_indices] == e2) & (symbols[j_indices] == e1)
            )

            pair_dists = distances[mask]

            # Histogram
            hist, _ = np.histogram(pair_dists, bins=self.rdf_bins, density=False)

            # Gaussian smearing (simple convolution approximation or just raw bins)
            # Here we stick to raw bins normalized by volume/number of atoms for robustness
            # Normalization: divide by total number of atoms to make it intensive
            hist = hist.astype(float) / len(atoms)

            for k, val in enumerate(hist):
                features[f"rdf_{e1}_{e2}_{k}"] = val

        return features

    def compute_local_env(self, atoms):
        """
        Computes BVS, ECoN, and Anisotropy per atom.
        """
        # Get neighbors up to cutoff
        # We need vectors for anisotropy
        i_indices, j_indices, distances, vectors = ase.neighborlist.neighbor_list(
            "ijdD", atoms, LOCAL_ENV_CUTOFF, self_interaction=False
        )

        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        # Initialize storage
        atom_bvs = np.zeros(n_atoms)
        atom_econ = np.zeros(n_atoms)
        atom_aniso = np.zeros(n_atoms)

        for idx in range(n_atoms):
            # Filter neighbors for this atom
            mask = i_indices == idx

            if not np.any(mask):
                continue

            nbs_dists = distances[mask]
            nbs_indices = j_indices[mask]
            nbs_vectors = vectors[mask]
            nbs_symbols = [symbols[j] for j in nbs_indices]

            # BVS
            atom_bvs[idx] = calculate_bvs(symbols[idx], nbs_symbols, nbs_dists)

            # ECoN
            atom_econ[idx] = calculate_econ(nbs_dists)

            # Anisotropy
            atom_aniso[idx] = calculate_local_anisotropy(nbs_vectors)

        return {"bvs": atom_bvs, "econ": atom_econ, "aniso": atom_aniso}

    def compute_polyhedral_connectivity(self, atoms):
        """
        Counts corner, edge, and face sharing for cations.
        """
        symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(atoms)

        # 1. Identify Metal-Oxygen bonds
        # Using a strict cutoff for bonding
        i_indices, j_indices = ase.neighborlist.neighbor_list(
            "ij", atoms, CONNECTIVITY_CUTOFF, self_interaction=False
        )

        # Build adjacency list: Metal Index -> Set of Oxygen Indices
        metal_oxygen_bonds = {
            i: set() for i in range(n_atoms) if symbols[i] in self.metals
        }

        for k in range(len(i_indices)):
            idx_i = i_indices[k]
            idx_j = j_indices[k]

            if symbols[idx_i] in self.metals and symbols[idx_j] == "O":
                metal_oxygen_bonds[idx_i].add(idx_j)

        # 2. Initialize counters per metal atom
        # Counts: how many OTHER metal atoms does this atom share X oxygens with?
        corner_counts = {i: 0 for i in metal_oxygen_bonds}
        edge_counts = {i: 0 for i in metal_oxygen_bonds}
        face_counts = {i: 0 for i in metal_oxygen_bonds}

        metal_indices = list(metal_oxygen_bonds.keys())

        # 3. Iterate pairs of metals
        for idx_a in range(len(metal_indices)):
            for idx_b in range(idx_a + 1, len(metal_indices)):
                m1 = metal_indices[idx_a]
                m2 = metal_indices[idx_b]

                # Intersection of oxygen neighbors
                shared_oxys = metal_oxygen_bonds[m1].intersection(
                    metal_oxygen_bonds[m2]
                )
                n_shared = len(shared_oxys)

                if n_shared == 1:
                    corner_counts[m1] += 1
                    corner_counts[m2] += 1
                elif n_shared == 2:
                    edge_counts[m1] += 1
                    edge_counts[m2] += 1
                elif n_shared >= 3:
                    face_counts[m1] += 1
                    face_counts[m2] += 1

        # Convert to arrays aligned with atom indices (fill non-metals with NaN or handle in aggregation)
        # We will return a dict mapping atom_index -> counts, but for aggregation we need arrays
        # Initialize full arrays with NaNs
        res_corner = np.full(n_atoms, np.nan)
        res_edge = np.full(n_atoms, np.nan)
        res_face = np.full(n_atoms, np.nan)

        for idx in metal_indices:
            res_corner[idx] = corner_counts[idx]
            res_edge[idx] = edge_counts[idx]
            res_face[idx] = face_counts[idx]

        return {"conn_corner": res_corner, "conn_edge": res_edge, "conn_face": res_face}

    def aggregate_distributions(self, atoms, local_props):
        """
        Aggregates atom-level properties into structure-level features using percentiles,
        grouped by element type.
        """
        symbols = np.array(atoms.get_chemical_symbols())
        features = {}
        percentiles = [0, 25, 50, 75, 100]

        # Properties to aggregate
        # local_props contains: bvs, econ, aniso, conn_corner, conn_edge, conn_face

        # Group by element
        for elem in self.all_elements:
            mask = symbols == elem

            # If element not present, we will get zeros from compute_percentiles

            for prop_name, prop_values in local_props.items():
                # For connectivity, only Metals have values (Oxygens are NaN)
                # If elem is O and prop is conn_*, values are all NaN, compute_percentiles handles empty/nan

                subset_vals = prop_values[mask]

                stats = compute_percentiles(subset_vals, percentiles)

                for p, val in zip(percentiles, stats):
                    features[f"{elem}_{prop_name}_p{p}"] = val

        # Also aggregate over ALL Metals combined (useful for general topology)
        metal_mask = np.isin(symbols, self.metals)
        for prop_name, prop_values in local_props.items():
            subset_vals = prop_values[metal_mask]
            stats = compute_percentiles(subset_vals, percentiles)
            for p, val in zip(percentiles, stats):
                features[f"Metal_{prop_name}_p{p}"] = val

        return features

    def featurize(self, atoms):
        # 1. Macroscopic
        feats = self.compute_macroscopic(atoms)

        # 2. RDF
        feats.update(self.compute_rdf(atoms))

        # 3. Local Environment
        local_env = self.compute_local_env(atoms)

        # 4. Connectivity
        connectivity = self.compute_polyhedral_connectivity(atoms)

        # Merge local properties for aggregation
        all_local_props = {**local_env, **connectivity}

        # 5. Aggregation
        dist_feats = self.aggregate_distributions(atoms, all_local_props)
        feats.update(dist_feats)

        return feats


def process_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata, computes features for each structure, and returns a DataFrame.
    Implements caching using Parquet files.
    """
    cache_file = os.path.join(WORKING_DIR, f"{split}_features_idea_40.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Computing features for {split} set...")

    # Load metadata
    # Use debug mode from config if needed, but the function signature doesn't take debug arg.
    # We rely on the global DEBUG_MODE in config if we were to pass it, but here we just follow instructions.
    # The prompt says "You must include hyperparameters to control dataset size (for debugging)".
    # We will use the DEBUG_MODE constant from config.py to control load_metadata.

    meta_df = load_metadata(split, debug=DEBUG_MODE, sample_size=DEBUG_SAMPLE_SIZE)

    featurizer = StructureFeaturizer()

    features_list = []

    # Iterate and featurize
    # Using a simple loop. For production, multiprocessing could be used, but keeping it simple and robust here.
    for idx, row in meta_df.iterrows():
        try:
            atoms = load_structure(row["file_path"])
            feats = featurizer.featurize(atoms)
            # Add ID to merge later if needed, though index alignment usually suffices
            feats["id"] = row["id"]
            features_list.append(feats)
        except Exception as e:
            print(f"Error processing id {row.get('id', 'unknown')}: {e}")
            # Append empty dict or None? Better to skip or fill with NaNs.
            # We'll append a dict with 'id' and NaNs for consistency is tricky without knowing columns.
            # We will skip and then re-index.
            continue

    if not features_list:
        raise RuntimeError("No features computed. Check input data.")

    feat_df = pd.DataFrame(features_list)

    # Merge with metadata (targets + lattice params)
    # We want to keep targets and original tabular features
    # meta_df has 'id'. feat_df has 'id'.
    merged_df = pd.merge(meta_df, feat_df, on="id", how="inner")

    # Save to cache
    print(f"Saving features to {cache_file}")
    merged_df.to_parquet(cache_file, index=False)

    return merged_df
