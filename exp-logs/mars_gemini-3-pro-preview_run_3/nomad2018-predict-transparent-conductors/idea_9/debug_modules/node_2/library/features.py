import os
import numpy as np
import pandas as pd
import ase.io
from library.config import Config
from library.utils import get_atomic_properties


def load_atoms(rel_path):
    """
    Loads atomic structure from an xyz file using ASE.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure object.
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    # ASE automatically handles the extended XYZ format with lattice vectors
    # Explicitly specify format='aims' as the files contain FHI-aims headers despite .xyz extension
    atoms = ase.io.read(full_path, format="aims")
    return atoms


def get_geometric_features(atoms):
    """
    Calculates explicit physical descriptors: Unit Cell Volume and Mass Density.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing 'geo_volume' and 'geo_density'.
    """
    vol = atoms.get_volume()
    # Sum of atomic masses in atomic mass units (AMU)
    mass = sum(atoms.get_masses())

    # Density calculation (AMU / Angstrom^3)
    # This is proportional to g/cm^3
    density = mass / vol if vol > 0 else 0.0

    return {"geo_volume": vol, "geo_density": density}


def get_chemical_disorder(atoms):
    """
    Calculates chemical disorder descriptors based on the variance of
    Pauling Electronegativity and Ionic Radii across the cation sublattice.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing variances of electronegativity and radius.
    """
    cation_ens = []
    cation_radii = []

    symbols = atoms.get_chemical_symbols()

    for s in symbols:
        # In these oxides, Oxygen is the anion. We focus on Al, Ga, In cations.
        if s != "O":
            try:
                props = get_atomic_properties(s)
                cation_ens.append(props["EN"])
                cation_radii.append(props["R"])
            except ValueError:
                # Handle unexpected elements gracefully if necessary
                continue

    if not cation_ens:
        return {"disorder_en_var": 0.0, "disorder_r_var": 0.0}

    # Calculate variance to capture the degree of disorder/mismatch
    en_var = np.var(cation_ens)
    r_var = np.var(cation_radii)

    return {"disorder_en_var": en_var, "disorder_r_var": r_var}


def get_electrostatic_fingerprints(atoms):
    """
    Computes statistical moments of the Coulomb Matrix off-diagonal elements.
    The Coulomb Matrix captures the global electrostatic energy landscape.
    Interaction term: C_ij = (Z_i * Z_j) / r_ij

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Mean, Std, and Max of the electrostatic interaction terms.
    """
    # Get atomic numbers (Z)
    z = atoms.get_atomic_numbers()

    # Calculate pairwise distances using Minimum Image Convention (MIC)
    # to account for periodic boundary conditions of the crystal.
    dists = atoms.get_all_distances(mic=True)

    # Handle diagonal (self-interaction distance is 0)
    # We set diagonal to infinity so that 1/r becomes 0
    np.fill_diagonal(dists, np.inf)

    # Compute Coulomb matrix: Z_i * Z_j / r_ij
    # Outer product computes Z_i * Z_j for all pairs
    charge_product = np.outer(z, z)
    coulomb_matrix = charge_product / dists

    # Extract off-diagonal elements (interactions)
    # We use the upper triangle to avoid duplicates and self-interactions (which are 0)
    rows, cols = np.triu_indices_from(coulomb_matrix, k=1)
    interactions = coulomb_matrix[rows, cols]

    if len(interactions) == 0:
        return {"electro_mean": 0.0, "electro_std": 0.0, "electro_max": 0.0}

    return {
        "electro_mean": np.mean(interactions),
        "electro_std": np.std(interactions),
        "electro_max": np.max(interactions),
    }


def process_single_entry(row):
    """
    Helper function to process a single row of metadata and extract all features.

    Args:
        row (pd.Series): A row from the metadata dataframe containing 'file_path'.

    Returns:
        dict: Combined feature dictionary.
    """
    try:
        atoms = load_atoms(row["file_path"])

        geo_feats = get_geometric_features(atoms)
        disorder_feats = get_chemical_disorder(atoms)
        electro_feats = get_electrostatic_fingerprints(atoms)

        # Merge all feature dictionaries
        features = {**geo_feats, **disorder_feats, **electro_feats}
        return features

    except Exception as e:
        print(f"Warning: Error processing ID {row.get('id', 'unknown')}: {e}")
        # Return zeroed features to maintain dataframe consistency
        return {
            "geo_volume": 0.0,
            "geo_density": 0.0,
            "disorder_en_var": 0.0,
            "disorder_r_var": 0.0,
            "electro_mean": 0.0,
            "electro_std": 0.0,
            "electro_max": 0.0,
        }


def generate_features(metadata_df, save_path, load_cached_data=True):
    """
    Main function to generate physics-informed features for a dataset.
    Implements caching mechanism using parquet files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata (must include 'file_path').
        save_path (str): Path to save/load the cached parquet file.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame with original metadata and new physics features.
    """
    # Ensure the directory for the cache file exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(save_path):
        print(f"Loading cached features from {save_path}")
        try:
            cached_df = pd.read_parquet(save_path)
            # Verify length matches (simple integrity check)
            if len(cached_df) == len(metadata_df):
                return cached_df
            else:
                print("Cache size mismatch. Recomputing features...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing features...")

    # 2. Compute features from scratch
    print(f"Generating physics-informed features for {len(metadata_df)} samples...")

    feature_list = []
    # Iterate through metadata and extract features for each material
    for _, row in metadata_df.iterrows():
        feats = process_single_entry(row)
        feature_list.append(feats)

    # Create DataFrame from list of dicts
    features_df = pd.DataFrame(feature_list)

    # 3. Combine with original metadata
    # We preserve original columns (like composition, spacegroup) as they are valuable tabular features
    # We drop 'file_path' as it is not a predictive feature
    meta_features = metadata_df.drop(columns=["file_path"], errors="ignore")

    # Align indices
    features_df.index = metadata_df.index

    final_df = pd.concat([meta_features, features_df], axis=1)

    # 4. Save to cache
    print(f"Saving features to {save_path}")
    final_df.to_parquet(save_path)

    return final_df
