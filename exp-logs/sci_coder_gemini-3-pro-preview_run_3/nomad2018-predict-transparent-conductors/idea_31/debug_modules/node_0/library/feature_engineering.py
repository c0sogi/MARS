import os
import numpy as np
import pandas as pd
import ase.io
from library.config import INPUT_DIR, WORKING_DIR
from library.descriptors_electrostatics import ElectrostaticsCalculator
from library.descriptors_geometry import GeometryCalculator


class FeaturePipeline:
    """
    Orchestrates the generation of Electro-Geometric Distributional Fingerprints (EGDF).
    Integrates Global Electrostatic, Local Chemo-Symmetry, and Geometric views.
    """

    def __init__(self):
        # Initialize calculators
        # Electrostatics: Madelung Energy, Bond Valence Sums (BVS), Bond Valence Vector Sums (BVVS)
        self.electro_calc = ElectrostaticsCalculator()

        # Geometry: Radial Distribution Functions (RDF), Effective Coordination Number (ECoN), Macroscopic props
        # Using cutoffs consistent with the EGDF strategy
        self.geo_calc = GeometryCalculator(
            rdf_cutoff=10.0, rdf_bins=50, econ_cutoff=3.5
        )

    def process_structure(self, file_path: str) -> dict:
        """
        Computes all features for a single structure file.

        Args:
            file_path (str): Path to the geometry.xyz file.

        Returns:
            dict: Dictionary of computed features.
        """
        try:
            atoms = ase.io.read(file_path)
        except Exception:
            # Return empty dict if file cannot be read
            return {}

        features = {}

        # 1. Global Electrostatic View
        # Madelung Energy
        try:
            features["madelung_energy"] = self.electro_calc.calculate_madelung_energy(
                atoms
            )
        except Exception:
            features["madelung_energy"] = np.nan

        # 2. Local Chemo-Symmetry View (BVS, BVVS, GII)
        # This includes distributional aggregation (percentiles) per element
        try:
            bvs_feats = self.electro_calc.calculate_bvs_features(atoms)
            features.update(bvs_feats)
        except Exception:
            pass

        # 3. Geometric View (RDF, ECoN, Macroscopic)
        # This includes RDF histograms and ECoN percentiles per element
        try:
            geo_feats = self.geo_calc.compute_features(atoms)
            features.update(geo_feats)
        except Exception:
            pass

        return features

    def generate_features(
        self, metadata_df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Generates features for the given metadata dataframe.
        Handles caching to parquet files to save time on subsequent runs.
        Merges computed features with the original tabular metadata.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: DataFrame containing original metadata merged with new features.
        """
        # Define cache path in the working directory
        cache_path = os.path.join(WORKING_DIR, f"{split_name}_egdf_features.parquet")

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        computed_df = None

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                computed_df = pd.read_parquet(cache_path)
            except Exception:
                # If load fails, we will recompute
                computed_df = None

        # 2. Compute if not loaded
        if computed_df is None:
            features_list = []

            for _, row in metadata_df.iterrows():
                struct_id = row["id"]
                rel_path = row["file_path"]
                full_path = os.path.join(INPUT_DIR, rel_path)

                if os.path.exists(full_path):
                    feats = self.process_structure(full_path)
                    feats["id"] = struct_id
                    features_list.append(feats)
                else:
                    # Handle missing file case by preserving ID (features will be NaN)
                    features_list.append({"id": struct_id})

            computed_df = pd.DataFrame(features_list)

            # Save to cache
            computed_df.to_parquet(cache_path, index=False)

        # 3. Merge with metadata
        # Ensure 'id' types match for merging
        metadata_df["id"] = metadata_df["id"].astype(int)
        computed_df["id"] = computed_df["id"].astype(int)

        # Merge computed features into the metadata dataframe
        merged_df = pd.merge(metadata_df, computed_df, on="id", how="left")

        return merged_df
