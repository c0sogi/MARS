import os
import numpy as np
import pandas as pd
from library.config import ATOMIC_MASSES, WORKING_DIR


class PhysicalFeaturizer:
    """
    Featurizer for extracting explicit physical descriptors from material metadata
    and geometry.
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_analytical_volume(df):
        """
        Calculates unit cell volume analytically from lattice parameters in the dataframe.
        V = abc * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))

        Args:
            df (pd.DataFrame): Dataframe containing lattice vectors and angles.

        Returns:
            np.ndarray: Array of calculated volumes.
        """
        # Extract lattice lengths
        a = df["lattice_vector_1_ang"].values
        b = df["lattice_vector_2_ang"].values
        c = df["lattice_vector_3_ang"].values

        # Extract and convert angles to radians
        alpha_rad = np.radians(df["lattice_angle_alpha_degree"].values)
        beta_rad = np.radians(df["lattice_angle_beta_degree"].values)
        gamma_rad = np.radians(df["lattice_angle_gamma_degree"].values)

        # Precompute cosines
        ca = np.cos(alpha_rad)
        cb = np.cos(beta_rad)
        cg = np.cos(gamma_rad)

        # Calculate volume factor
        term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg

        # Ensure non-negative for sqrt (handle potential floating point errors near zero)
        term = np.maximum(term, 0)

        volume = a * b * c * np.sqrt(term)
        return volume

    @staticmethod
    def calculate_density(atoms_list, volumes):
        """
        Calculates density given ASE atoms objects and pre-calculated volumes.
        Density = Total Mass / Volume

        Args:
            atoms_list (list): List of ASE Atoms objects.
            volumes (np.ndarray): Array of unit cell volumes.

        Returns:
            np.ndarray: Array of calculated densities.
        """
        densities = []
        for i, atoms in enumerate(atoms_list):
            vol = volumes[i]

            # Handle edge case of zero or extremely small volume
            if vol <= 1e-6:
                densities.append(0.0)
                continue

            # Calculate total mass of the unit cell
            total_mass = 0.0
            symbols = atoms.get_chemical_symbols()
            for sym in symbols:
                # Use provided atomic masses constant
                total_mass += ATOMIC_MASSES.get(sym, 0.0)

            # Density in amu / Angstrom^3
            density = total_mass / vol
            densities.append(density)

        return np.array(densities)

    def featurize(self, metadata_df, atoms_list, split_name, load_cached_data=True):
        """
        Generates physical features for a given dataset split.

        Args:
            metadata_df (pd.DataFrame): Metadata containing lattice params and composition.
            atoms_list (list): List of ASE Atoms objects corresponding to metadata_df rows.
            split_name (str): 'train', 'val', or 'test' for caching purposes.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Dataframe of physical features.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        cache_path = os.path.join(
            WORKING_DIR, f"{split_name}_physical_features.parquet"
        )

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading physical features from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing features.")

        print(f"Generating physical features for {split_name}...")

        # 2. Compute Features
        # Calculate Volume analytically (more robust than parsing from XYZ comments)
        volumes = self.calculate_analytical_volume(metadata_df)

        # Calculate Density using computed volume and atomic masses
        densities = self.calculate_density(atoms_list, volumes)

        # 3. Construct Feature DataFrame
        features = pd.DataFrame()
        features["analytical_volume"] = volumes
        features["density"] = densities

        # Copy relevant explicit features from metadata
        cols_to_copy = [
            "spacegroup",
            "number_of_total_atoms",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
        ]

        for col in cols_to_copy:
            if col in metadata_df.columns:
                features[col] = metadata_df[col].values

        # 4. Save to cache
        try:
            features.to_parquet(cache_path, index=False)
            print(f"Saved physical features to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save physical features cache: {e}")

        return features
