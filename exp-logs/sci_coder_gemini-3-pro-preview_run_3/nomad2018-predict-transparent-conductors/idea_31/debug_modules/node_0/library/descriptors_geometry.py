import os
import numpy as np
import pandas as pd
import ase.io
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import INPUT_DIR, WORKING_DIR


class GeometryCalculator:
    """
    Calculates geometric and structural descriptors including:
    - Macroscopic properties (volume, density, lattice parameters)
    - Effective Coordination Number (ECoN) distributions per element
    - Element-resolved Radial Distribution Functions (RDF)
    """

    def __init__(self, rdf_cutoff=10.0, rdf_bins=50, econ_cutoff=3.5):
        self.rdf_cutoff = rdf_cutoff
        self.rdf_bins = rdf_bins
        self.econ_cutoff = econ_cutoff
        self.elements = ["Al", "Ga", "In", "O"]

    def get_macroscopic_props(self, atoms: Atoms) -> dict:
        """
        Extracts unit cell volume, density, and lattice parameters.
        """
        feats = {}

        # Volume and Density
        vol = atoms.get_volume()
        mass = sum(atoms.get_masses())
        feats["geo_volume"] = vol
        feats["geo_density"] = mass / vol if vol > 1e-5 else 0.0
        feats["geo_vol_per_atom"] = vol / len(atoms) if len(atoms) > 0 else 0.0

        # Lattice Parameters
        lengths, angles = atoms.get_cell_lengths_and_angles()
        feats["geo_lat_a"] = lengths[0]
        feats["geo_lat_b"] = lengths[1]
        feats["geo_lat_c"] = lengths[2]
        feats["geo_lat_alpha"] = angles[0]
        feats["geo_lat_beta"] = angles[1]
        feats["geo_lat_gamma"] = angles[2]

        return feats

    def calculate_econ(self, atoms: Atoms) -> dict:
        """
        Calculates Effective Coordination Number (ECoN) for each atom and
        aggregates them into percentiles per element.

        Uses a simplified Hoppe-like weighting:
        ECoN_i = sum_j exp(1 - (r_ij / r_av_i)^6)
        where r_av_i is the average bond length of the nearest neighbors.
        """
        feats = {}
        n_atoms = len(atoms)
        symbols = np.array(atoms.get_chemical_symbols())

        # Get neighbors within a generous cutoff to include first shell
        # 3.5 Angstrom covers M-O and O-M bonds well in these oxides
        i_idx, j_idx, d_vals = neighbor_list("ijd", atoms, self.econ_cutoff)

        # Initialize ECoN array
        econ_values = np.zeros(n_atoms)

        # Process each atom
        for i in range(n_atoms):
            # Get distances for neighbors of atom i
            dists = d_vals[i_idx == i]

            if len(dists) == 0:
                econ_values[i] = 0.0
                continue

            # Filter to "bonding" neighbors for r_av calculation
            # A simple heuristic: take neighbors within 1.2 * min_dist
            # This avoids including second shell in the average distance calculation
            min_dist = np.min(dists)
            bonding_dists = dists[dists <= (1.2 * min_dist + 0.1)]

            if len(bonding_dists) == 0:
                r_av = min_dist  # Fallback
            else:
                r_av = np.mean(bonding_dists)

            # Calculate weights for all neighbors within cutoff
            # weight = exp(1 - (r / r_av)^6)
            # This weight drops rapidly for r > r_av
            weights = np.exp(1.0 - (dists / r_av) ** 6)

            # ECoN is sum of weights
            econ_values[i] = np.sum(weights)

        # Aggregate per element
        percentiles = [0, 25, 50, 75, 100]
        for elem in self.elements:
            mask = symbols == elem
            if np.sum(mask) > 0:
                vals = econ_values[mask]
                p_vals = np.percentile(vals, percentiles)
                for p, v in zip(percentiles, p_vals):
                    feats[f"geo_ECoN_{elem}_p{p}"] = v
            else:
                for p in percentiles:
                    feats[f"geo_ECoN_{elem}_p{p}"] = np.nan

        return feats

    def calculate_rdf(self, atoms: Atoms) -> dict:
        """
        Calculates Element-Resolved Radial Distribution Functions.

        For each element type, computes the distribution of distances to all other atoms.
        """
        feats = {}
        symbols = np.array(atoms.get_chemical_symbols())
        n_atoms = len(atoms)

        # Get all pairwise distances up to rdf_cutoff
        # We use neighbor_list to handle PBC correctly
        i_idx, j_idx, d_vals = neighbor_list("ijd", atoms, self.rdf_cutoff)

        # Bins
        bins = np.linspace(0, self.rdf_cutoff, self.rdf_bins + 1)

        for elem in self.elements:
            # Indices of atoms of this element
            elem_indices = np.where(symbols == elem)[0]

            if len(elem_indices) == 0:
                # Fill with zeros if element not present
                for b in range(self.rdf_bins):
                    feats[f"geo_RDF_{elem}_bin{b}"] = 0.0
                continue

            # Filter distances where the central atom (i) is of type elem
            # np.isin is useful here
            mask = np.isin(i_idx, elem_indices)
            elem_dists = d_vals[mask]

            # Compute histogram
            hist, _ = np.histogram(elem_dists, bins=bins)

            # Normalize by number of atoms of this element
            # This gives "average number of neighbors at distance r"
            hist = hist.astype(float) / len(elem_indices)

            # Store features
            for b, count in enumerate(hist):
                feats[f"geo_RDF_{elem}_bin{b}"] = count

        return feats

    def compute_features(self, atoms: Atoms) -> dict:
        """
        Main entry point to compute all geometric features.
        """
        feats = {}
        feats.update(self.get_macroscopic_props(atoms))
        feats.update(self.calculate_econ(atoms))
        feats.update(self.calculate_rdf(atoms))
        return feats


def extract_geometry_features(
    metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Extracts geometric descriptors for a dataset.
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

    cache_path = os.path.join(WORKING_DIR, f"{split_name}_geometry_features.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometry features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute features
    print(f"Computing geometry features for {split_name}...")

    calculator = GeometryCalculator(rdf_cutoff=10.0, rdf_bins=40, econ_cutoff=4.0)
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
            # Add entry with just ID to maintain alignment (will result in NaNs which XGBoost handles)
            features_list.append({"id": struct_id})

    # 3. Create DataFrame
    df_features = pd.DataFrame(features_list)

    # 4. Save to cache
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved geometry features to {cache_path}")

    return df_features
