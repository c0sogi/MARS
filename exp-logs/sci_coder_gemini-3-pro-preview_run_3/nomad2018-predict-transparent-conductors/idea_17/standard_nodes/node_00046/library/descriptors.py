import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from tqdm import tqdm

from library.config import Config


class FeatureExtractor:
    """
    Implements Geometric Fingerprinting.
    Extracts Physical, RDF, and CR-LEM features.
    Cite solution_lesson_node_00044: Removed fragile GNN dependencies.
    """

    def __init__(self):
        # RDF Parameters
        self.rdf_cutoff = Config.RDF_CUTOFF
        self.rdf_bins = Config.RDF_NUM_BINS
        self.rdf_r = np.linspace(0, self.rdf_cutoff, self.rdf_bins + 1)

        # Species mapping for specific features
        self.cation_symbols = {"Al", "Ga", "In"}
        self.anion_symbols = {"O"}
        self.all_symbols = ["Al", "Ga", "In", "O"]

    def get_physical_properties(self, atoms):
        """
        Computes basic physical properties: Volume and Density.
        """
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        density = mass / vol if vol > 1e-6 else 0.0
        return {"phys_volume": vol, "phys_density": density, "phys_natoms": len(atoms)}

    def compute_rdf(self, atoms):
        """
        Computes Element-Resolved Radial Distribution Functions.
        Normalized by the number of atoms to ensure size-invariance.
        """
        features = {}

        # Get all pairwise distances
        # We use a slightly larger cutoff to ensure we cover the bins
        i_indices, j_indices, dists = neighbor_list("ijd", atoms, self.rdf_cutoff)

        symbols = atoms.get_chemical_symbols()

        # Iterate over all unique pairs of elements to create specific RDFs
        # Al-Al, Al-Ga, ..., O-O
        # To keep order consistent, we use the sorted list self.all_symbols

        for idx1, el1 in enumerate(self.all_symbols):
            for idx2, el2 in enumerate(self.all_symbols):
                if idx1 > idx2:
                    continue  # Symmetric

                pair_label = f"rdf_{el1}_{el2}"

                # Filter distances for this pair
                # We need to match i_indices to el1 and j_indices to el2 (or vice versa)

                mask_i_el1 = np.array([symbols[i] == el1 for i in i_indices])
                mask_j_el2 = np.array([symbols[j] == el2 for j in j_indices])

                # Case 1: i is el1, j is el2
                mask1 = mask_i_el1 & mask_j_el2

                # Case 2: i is el2, j is el1 (if different elements)
                if el1 != el2:
                    mask_i_el2 = np.array([symbols[i] == el2 for i in i_indices])
                    mask_j_el1 = np.array([symbols[j] == el1 for j in j_indices])
                    mask2 = mask_i_el2 & mask_j_el1
                    mask = mask1 | mask2
                else:
                    mask = mask1

                pair_dists = dists[mask]

                # Compute Histogram
                hist, _ = np.histogram(pair_dists, bins=self.rdf_r, density=False)

                # Normalize by total atoms to make it intensive-like
                norm_hist = hist / len(atoms)

                # Store features
                for b in range(self.rdf_bins):
                    features[f"{pair_label}_bin_{b}"] = norm_hist[b]

        return features

    def compute_cr_lem(self, atoms):
        """
        Computes Chemically-Resolved Local Environment Moments.
        Mean and Std of Coordination Number (CN) and Bond Angle Variance.
        """
        features = {}
        cutoff = Config.LEM_CUTOFF

        # Neighbor list for local environment
        # We need full connectivity for angles
        nl = neighbor_list("ijD", atoms, cutoff)
        i_idx, j_idx, D_vecs = nl

        symbols = np.array(atoms.get_chemical_symbols())
        unique_elements = self.all_symbols

        # Pre-compute per-atom metrics
        atom_cns = np.zeros(len(atoms))
        atom_angle_vars = np.zeros(len(atoms))

        # Count neighbors
        unique, counts = np.unique(i_idx, return_counts=True)
        atom_cns[unique] = counts

        # Compute angles for atoms with > 1 neighbor
        # This is computationally intensive, so we do a simplified variance estimation
        # or iterate only over atoms with neighbors.
        # Vectorized angle calculation is tricky with variable neighbor counts.
        # We iterate over atoms for angles.

        for a in range(len(atoms)):
            neighbors = j_idx[i_idx == a]
            if len(neighbors) < 2:
                atom_angle_vars[a] = 0.0
                continue

            # Get vectors
            vecs = D_vecs[i_idx == a]
            # Normalize
            norms = np.linalg.norm(vecs, axis=1)
            # Avoid division by zero
            norms[norms < 1e-9] = 1.0
            vecs_norm = vecs / norms[:, None]

            # Compute cosine similarity matrix for neighbors
            # (N_neigh, 3) @ (3, N_neigh) -> (N_neigh, N_neigh)
            cos_angles = vecs_norm @ vecs_norm.T

            # We only care about off-diagonal (actual angles)
            # Extract upper triangle
            mask = np.triu(np.ones_like(cos_angles, dtype=bool), k=1)
            valid_cos = cos_angles[mask]

            # Convert to angles (radians)
            # Clip for numerical stability
            valid_cos = np.clip(valid_cos, -1.0, 1.0)
            angles = np.arccos(valid_cos)

            if len(angles) > 0:
                atom_angle_vars[a] = np.var(angles)
            else:
                atom_angle_vars[a] = 0.0

        # Aggregate by element
        for el in unique_elements:
            mask = symbols == el
            if np.sum(mask) > 0:
                cns = atom_cns[mask]
                avs = atom_angle_vars[mask]
                features[f"lem_{el}_cn_mean"] = np.mean(cns)
                features[f"lem_{el}_cn_std"] = np.std(cns)
                features[f"lem_{el}_av_mean"] = np.mean(avs)
                features[f"lem_{el}_av_std"] = np.std(avs)
            else:
                features[f"lem_{el}_cn_mean"] = 0.0
                features[f"lem_{el}_cn_std"] = 0.0
                features[f"lem_{el}_av_mean"] = 0.0
                features[f"lem_{el}_av_std"] = 0.0

        return features

    def process_structure(self, atoms):
        """
        Master function to extract all features for a single structure.
        """
        # Physical
        phys_feats = self.get_physical_properties(atoms)

        # RDF
        rdf_feats = self.compute_rdf(atoms)

        # CR-LEM
        lem_feats = self.compute_cr_lem(atoms)

        # Combine
        all_feats = {**phys_feats, **rdf_feats, **lem_feats}
        return all_feats


def generate_features(metadata_df, save_path, load_cached_data=True):
    """
    Generates features for the given metadata dataframe.
    Handles caching via Parquet files.
    """
    # Check cache
    if load_cached_data and os.path.exists(save_path):
        print(f"Loading cached features from {save_path}...")
        return pd.read_parquet(save_path)

    print(f"Generating features for {len(metadata_df)} samples...")

    # Initialize extractor
    extractor = FeatureExtractor()

    features_list = []
    ids = []

    # Limit for debugging if configured
    if Config.SAMPLE_SIZE and Config.SAMPLE_SIZE < len(metadata_df):
        print(f"Debugging: processing only first {Config.SAMPLE_SIZE} samples.")
        metadata_df = metadata_df.iloc[: Config.SAMPLE_SIZE]

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        material_id = row["id"]

        try:
            # Cite debug_lesson_3: Explicitly specify file format
            atoms = ase.io.read(file_path, format="aims")
            feats = extractor.process_structure(atoms)
            feats["id"] = material_id
            features_list.append(feats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Append empty/nan row to maintain alignment if needed, or skip
            # Here we skip, but in production we might want to handle it
            continue

    # Create DataFrame
    if not features_list:
        raise ValueError("No features were generated. Check input paths.")

    features_df = pd.DataFrame(features_list)

    # Merge with original metadata to keep targets and tabular info
    # We merge on 'id'
    # First drop 'id' from metadata if it's the index, or ensure it's a column
    result_df = pd.merge(metadata_df, features_df, on="id", how="inner")

    # Save cache
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_parquet(save_path, index=False)
    print(f"Features saved to {save_path}")

    return result_df
