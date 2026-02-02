import os
import numpy as np
import pandas as pd
from library.config import (
    BVS_R0,
    BVS_B,
    RDF_CUTOFF,
    RDF_BINS,
    NEIGHBOR_CUTOFF,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    RANDOM_SEED,
)
from library.structure_utils import load_xyz, get_neighbor_list, get_cell_parameters
from ase.data import chemical_symbols, atomic_masses

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class Aggregator:
    """
    Aggregates atom-wise properties into global features using percentiles and moments.
    """

    def __init__(self, elements=["Al", "Ga", "In", "O"]):
        self.elements = elements
        self.stats = ["min", "25%", "50%", "75%", "max", "mean", "std"]

    def aggregate(self, values, atom_symbols, prefix):
        """
        Aggregates values by element type.

        Args:
            values (np.array): Array of values corresponding to atoms.
            atom_symbols (np.array): Array of chemical symbols for each atom.
            prefix (str): Prefix for feature names.

        Returns:
            dict: Dictionary of aggregated features.
        """
        features = {}

        # Global aggregation
        if len(values) > 0:
            features.update(self._compute_stats(values, f"{prefix}_global"))
        else:
            features.update(self._get_empty_stats(f"{prefix}_global"))

        # Element-wise aggregation
        for el in self.elements:
            mask = atom_symbols == el
            el_values = values[mask]
            if len(el_values) > 0:
                features.update(self._compute_stats(el_values, f"{prefix}_{el}"))
            else:
                features.update(self._get_empty_stats(f"{prefix}_{el}"))

        return features

    def _compute_stats(self, data, name):
        res = {}
        res[f"{name}_min"] = np.min(data)
        res[f"{name}_25%"] = np.percentile(data, 25)
        res[f"{name}_50%"] = np.median(data)
        res[f"{name}_75%"] = np.percentile(data, 75)
        res[f"{name}_max"] = np.max(data)
        res[f"{name}_mean"] = np.mean(data)
        res[f"{name}_std"] = np.std(data)
        return res

    def _get_empty_stats(self, name):
        res = {}
        for stat in self.stats:
            res[f"{name}_{stat}"] = 0.0
        return res


class BondValenceCalculator:
    """
    Computes Bond Valence Sums (BVS) for cations and anions.
    """

    def __init__(self, r0_params=BVS_R0, b_param=BVS_B):
        self.r0 = r0_params
        self.b = b_param

    def compute(self, atoms, i_indices, j_indices, distances):
        symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(atoms)
        bvs_values = np.zeros(n_atoms)

        # Iterate over all bonds
        for k in range(len(distances)):
            idx_i = i_indices[k]
            idx_j = j_indices[k]
            dist = distances[k]

            sym_i = symbols[idx_i]
            sym_j = symbols[idx_j]

            # Identify cation and anion
            cation = None
            if sym_i in self.r0 and sym_j == "O":
                cation = sym_i
            elif sym_j in self.r0 and sym_i == "O":
                cation = sym_j

            if cation:
                r0 = self.r0[cation]
                val = np.exp((r0 - dist) / self.b)
                # Add valence contribution to both atoms involved in the bond
                # (Conceptually BVS is for the ion, we sum contributions from neighbors)
                bvs_values[idx_i] += val

        # For anions (Oxygen), the BVS is the sum of bond valences from surrounding cations.
        # For cations, it is the sum of bond valences to surrounding anions.
        # The loop above accumulates these correctly for both i and j.

        return bvs_values


class GeometricDescriptor:
    """
    Computes Effective Coordination Number (ECoN) and Bond Angle Variance.
    """

    def compute(self, atoms, i_indices, j_indices, distances, vectors):
        n_atoms = len(atoms)
        symbols = np.array(atoms.get_chemical_symbols())

        # 1. Effective Coordination Number (ECoN)
        # Using a simplified exponential decay weight for robustness: w_ij = exp(1 - (d_ij / d_avg_nn)^6) is common,
        # but here we stick to a simple count within NEIGHBOR_CUTOFF as defined by the neighbor list generation,
        # weighted by inverse distance squared to give more weight to closer atoms.
        # ECoN_i = sum(1) is just coordination number. Let's use a continuous measure.
        # ECoN_i = sum(exp(-(d/d0)^2)) is arbitrary without d0.
        # Let's use the standard neighbor count from the provided cutoff list as a base geometric property.

        coord_numbers = np.zeros(n_atoms)
        # Count neighbors
        for idx in i_indices:
            coord_numbers[idx] += 1

        # 2. Bond Angle Variance & Topology (M-O-M angles)
        # We need to group vectors by central atom
        # Pre-allocate lists for neighbors
        neighbors = [[] for _ in range(n_atoms)]
        for k, idx_i in enumerate(i_indices):
            neighbors[idx_i].append(vectors[k])

        angle_variances = np.zeros(n_atoms)
        mom_angles = []  # Metal-Oxygen-Metal angles

        for i in range(n_atoms):
            vecs = np.array(neighbors[i])
            if len(vecs) < 2:
                angle_variances[i] = 0.0
                continue

            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1)
            # Avoid division by zero
            norms[norms < 1e-6] = 1.0
            unit_vecs = vecs / norms[:, np.newaxis]

            # Compute all unique pairs of angles
            angles = []
            n_neigh = len(unit_vecs)
            for a in range(n_neigh):
                for b in range(a + 1, n_neigh):
                    dot_prod = np.dot(unit_vecs[a], unit_vecs[b])
                    # Clip for numerical stability
                    dot_prod = np.clip(dot_prod, -1.0, 1.0)
                    angle = np.arccos(dot_prod) * (180.0 / np.pi)
                    angles.append(angle)

            if angles:
                angle_variances[i] = np.var(angles)

                # If central atom is Oxygen, these are M-O-M angles (assuming neighbors are Metals)
                # In this dataset, O is only bonded to metals within cutoff usually.
                if symbols[i] == "O":
                    mom_angles.extend(angles)

        return coord_numbers, angle_variances, np.array(mom_angles)


class RDFDescriptor:
    """
    Computes Element-Resolved Radial Distribution Functions.
    """

    def __init__(self, cutoff=RDF_CUTOFF, bins=RDF_BINS):
        self.cutoff = cutoff
        self.bins = bins
        self.pairs = [("Al", "O"), ("Ga", "O"), ("In", "O"), ("O", "O")]  # Key pairs

    def compute(self, atoms, i_indices, j_indices, distances):
        symbols = np.array(atoms.get_chemical_symbols())
        volume = atoms.get_volume()
        rdf_features = {}

        # Pre-compute masks for efficiency
        atom_types = set(symbols)

        for el1, el2 in self.pairs:
            # Check if elements exist in structure
            if el1 not in atom_types or el2 not in atom_types:
                hist = np.zeros(self.bins)
            else:
                # Filter distances for this pair
                # Mask for i being el1 and j being el2
                mask_i = symbols[i_indices] == el1
                mask_j = symbols[j_indices] == el2
                pair_mask = mask_i & mask_j

                pair_dists = distances[pair_mask]

                # Compute histogram
                hist, _ = np.histogram(
                    pair_dists, bins=self.bins, range=(0, self.cutoff)
                )

                # Normalize by volume and number of atoms to make it intensive-like
                # Standard RDF normalization is complex, here we use simple density normalization
                if len(pair_dists) > 0:
                    hist = hist.astype(float) / volume

            # Add to features
            for b in range(self.bins):
                rdf_features[f"RDF_{el1}_{el2}_{b}"] = hist[b]

        return rdf_features


class MacroscopicDescriptor:
    """
    Computes macroscopic properties: Volume, Density, Composition.
    """

    def compute(self, atoms):
        cell_params = get_cell_parameters(atoms)
        vol = cell_params["volume"]

        masses = atoms.get_masses()
        total_mass = sum(masses)
        density = total_mass / vol if vol > 0 else 0.0

        # Atomic composition fractions
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(symbols)
        comp = {}
        for el in ["Al", "Ga", "In", "O"]:
            comp[f"frac_{el}"] = symbols.count(el) / n_atoms

        return {
            "vol_per_atom": vol / n_atoms,
            "density": density,
            **comp,
            **{
                k: v for k, v in cell_params.items() if k != "volume"
            },  # Add a, b, c, alpha...
        }


def extract_features_from_atoms(atoms):
    """
    Orchestrates the feature extraction for a single structure.
    """
    # Initialize calculators
    aggregator = Aggregator()
    bvs_calc = BondValenceCalculator()
    geo_calc = GeometricDescriptor()
    rdf_calc = RDFDescriptor()
    macro_calc = MacroscopicDescriptor()

    # 1. Macroscopic Features
    features = macro_calc.compute(atoms)

    # 2. Neighbor List (computed once for BVS and Geo)
    # Use NEIGHBOR_CUTOFF for local chemistry/geometry
    idx_i, idx_j, dists, vecs = get_neighbor_list(atoms, cutoff=NEIGHBOR_CUTOFF)

    # 3. BVS Features
    bvs_vals = bvs_calc.compute(atoms, idx_i, idx_j, dists)
    symbols = np.array(atoms.get_chemical_symbols())
    features.update(aggregator.aggregate(bvs_vals, symbols, "BVS"))

    # 4. Geometric Features
    cn_vals, ang_var_vals, mom_angles = geo_calc.compute(
        atoms, idx_i, idx_j, dists, vecs
    )
    features.update(aggregator.aggregate(cn_vals, symbols, "CN"))
    features.update(aggregator.aggregate(ang_var_vals, symbols, "AngVar"))

    # Topological (M-O-M angles) aggregation
    if len(mom_angles) > 0:
        mom_stats = aggregator._compute_stats(mom_angles, "MOM_Angle")
    else:
        mom_stats = aggregator._get_empty_stats("MOM_Angle")
    features.update(mom_stats)

    # 5. RDF Features
    # Use RDF_CUTOFF for long-range interactions
    idx_i_rdf, idx_j_rdf, dists_rdf, _ = get_neighbor_list(atoms, cutoff=RDF_CUTOFF)
    rdf_feats = rdf_calc.compute(atoms, idx_i_rdf, idx_j_rdf, dists_rdf)
    features.update(rdf_feats)

    return features


def process_dataset(metadata_df):
    """
    Iterates over the metadata DataFrame and computes features for each structure.
    """
    data_list = []

    for _, row in metadata_df.iterrows():
        file_path = row["file_path"]
        try:
            atoms = load_xyz(file_path)
            feats = extract_features_from_atoms(atoms)
            feats["id"] = row["id"]  # Ensure ID is preserved
            data_list.append(feats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # In case of error, we might skip or append empty.
            # For robustness, we skip and let the merger handle missing IDs if any.
            continue

    return pd.DataFrame(data_list)


def generate_features(data_type="train", load_cached_data=True):
    """
    Main function to generate or load features.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet.

    Returns:
        pd.DataFrame: Feature matrix (X)
        pd.DataFrame: Target vector (y) if available, else None
    """
    # Determine paths based on data_type
    if data_type == "train":
        meta_path = TRAIN_METADATA_PATH
        feat_path = TRAIN_FEATURES_PATH
    elif data_type == "val":
        meta_path = VAL_METADATA_PATH
        feat_path = VAL_FEATURES_PATH
    elif data_type == "test":
        meta_path = TEST_METADATA_PATH
        feat_path = TEST_FEATURES_PATH
    else:
        raise ValueError("Invalid data_type")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(feat_path):
        print(f"Loading cached features from {feat_path}")
        df_features = pd.read_parquet(feat_path)

        # Load metadata to get targets
        df_meta = pd.read_csv(meta_path)

        # Merge to ensure alignment (though parquet should be self-contained if saved correctly)
        # We assume the parquet contains 'id' and all features.
        # We join targets from metadata.
        if data_type in ["train", "val"]:
            # Merge targets
            df_merged = df_features.merge(
                df_meta[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]],
                on="id",
                how="left",
            )

            # Separate X and y
            y = df_merged[["formation_energy_ev_natom", "bandgap_energy_ev"]]
            X = df_merged.drop(
                columns=["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
            )
            return X, y
        else:
            X = df_features.drop(columns=["id"])
            return X, None

    # 2. Compute from Scratch
    print(f"Computing features for {data_type} set...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    df_features = process_dataset(df_meta)

    # Save to cache
    print(f"Saving features to {feat_path}")
    df_features.to_parquet(feat_path, index=False)

    # Return X, y
    if data_type in ["train", "val"]:
        df_merged = df_features.merge(
            df_meta[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]],
            on="id",
            how="left",
        )
        y = df_merged[["formation_energy_ev_natom", "bandgap_energy_ev"]]
        X = df_merged.drop(
            columns=["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        )
        return X, y
    else:
        X = df_features.drop(columns=["id"])
        return X, None
