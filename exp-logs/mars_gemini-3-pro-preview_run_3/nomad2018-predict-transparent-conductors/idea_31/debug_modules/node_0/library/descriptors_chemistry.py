import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import INPUT_DIR, WORKING_DIR, BVS_R0, BVS_B, OXIDATION_STATES


class BondValenceCalculator:
    """
    Calculates Bond Valence Sum (BVS) and Bond Valence Vector Sum (BVVS) descriptors.
    These descriptors capture local chemical strain and directional symmetry breaking.
    """

    def __init__(self, cutoff: float = 6.0):
        self.cutoff = cutoff

    def _calculate_atomic_metrics(self, atoms: Atoms):
        """
        Computes BVS and BVVS vector for each atom.

        Returns:
            bvs (np.ndarray): Bond valence sum for each atom.
            bvvs (np.ndarray): Bond valence vector sum (magnitude) for each atom.
        """
        n_atoms = len(atoms)
        symbols = atoms.get_chemical_symbols()

        # Initialize arrays
        bvs = np.zeros(n_atoms)
        bvvs_vec = np.zeros((n_atoms, 3))

        # Get neighbors
        # i: central atom index, j: neighbor index, D: vector from i to j
        i_indices, j_indices, d_vectors = neighbor_list("ijD", atoms, self.cutoff)
        distances = np.linalg.norm(d_vectors, axis=1)

        for k, (i, j) in enumerate(zip(i_indices, j_indices)):
            el_i = symbols[i]
            el_j = symbols[j]
            dist = distances[k]

            # Check if bond parameters exist for this pair
            # We typically look for cation-anion pairs in oxides
            key = (el_i, el_j)

            if key in BVS_R0 and dist > 0.1:
                r0 = BVS_R0[key]
                # Bond valence s_ij = exp((R0 - R_ij) / b)
                s_ij = np.exp((r0 - dist) / BVS_B)

                # Accumulate BVS
                bvs[i] += s_ij

                # Accumulate BVVS (vector sum)
                # Direction vector from i to j is d_vectors[k]
                # Normalized direction: d_vectors[k] / dist
                # We weight the direction by the bond valence s_ij
                bvvs_vec[i] += s_ij * (d_vectors[k] / dist)

        # Compute magnitude of BVVS vectors
        bvvs_mag = np.linalg.norm(bvvs_vec, axis=1)

        return bvs, bvvs_mag

    def compute_features(self, atoms: Atoms) -> dict:
        """
        Aggregates atomic BVS and BVVS into structure-level distributional features.
        """
        features = {}
        symbols = np.array(atoms.get_chemical_symbols())
        unique_elements = ["Al", "Ga", "In", "O"]

        # Calculate atomic metrics
        bvs, bvvs = self._calculate_atomic_metrics(atoms)

        # 1. Global Instability Index (GII)
        # GII = sqrt( sum( (V_calc - V_ideal)^2 ) / N )
        ideal_valences = np.array([abs(OXIDATION_STATES.get(s, 0.0)) for s in symbols])
        # Avoid division by zero if atoms is empty (unlikely)
        if len(atoms) > 0:
            gii = np.sqrt(np.mean((bvs - ideal_valences) ** 2))
        else:
            gii = 0.0
        features["chem_GII"] = gii

        # 2. Distributional Features per Element
        percentiles = [0, 25, 50, 75, 100]

        for elem in unique_elements:
            mask = symbols == elem
            if np.sum(mask) > 0:
                elem_bvs = bvs[mask]
                elem_bvvs = bvvs[mask]

                # BVS Percentiles
                p_bvs = np.percentile(elem_bvs, percentiles)
                for p, val in zip(percentiles, p_bvs):
                    features[f"chem_BVS_{elem}_p{p}"] = val

                # BVVS Percentiles
                p_bvvs = np.percentile(elem_bvvs, percentiles)
                for p, val in zip(percentiles, p_bvvs):
                    features[f"chem_BVVS_{elem}_p{p}"] = val
            else:
                # Fill with NaN if element is not present in the structure
                for p in percentiles:
                    features[f"chem_BVS_{elem}_p{p}"] = np.nan
                    features[f"chem_BVVS_{elem}_p{p}"] = np.nan

        return features


def extract_chemistry_features(
    metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Extracts chemistry descriptors (BVS, BVVS, GII) for a dataset.
    Implements caching using parquet files in the working directory.

    Args:
        metadata_df: DataFrame containing 'id' and 'file_path'.
        split_name: Name of the split (e.g., 'train', 'val', 'test').
        load_cached_data: Whether to load from cache if available.

    Returns:
        pd.DataFrame: Features indexed by 'id'.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(WORKING_DIR, f"{split_name}_chemistry_features.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached chemistry features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute features
    print(f"Computing chemistry features for {split_name}...")

    calculator = BondValenceCalculator(cutoff=6.0)
    features_list = []

    for _, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        struct_id = row["id"]

        try:
            atoms = ase.io.read(file_path)
            feats = calculator.compute_features(atoms)
            feats["id"] = struct_id
            features_list.append(feats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Add entry with NaN features but correct ID to maintain alignment
            # (Though in this competition data is usually clean)
            empty_feats = {"id": struct_id}
            features_list.append(empty_feats)

    # 3. Create DataFrame
    df_features = pd.DataFrame(features_list)

    # Ensure 'id' is the first column or index for merging later
    # We keep it as a column for pd.merge

    # 4. Save to cache
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved chemistry features to {cache_path}")

    return df_features
