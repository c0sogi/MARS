import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import ATOMIC_PROPERTIES, NEIGHBOR_CUTOFF, INPUT_DIR, WORKING_DIR


def read_structure(rel_path):
    """
    Reads a geometry file using ASE.

    Args:
        rel_path (str): Relative path to the geometry file from INPUT_DIR.

    Returns:
        ase.Atoms: The atomic structure object, or None if reading fails.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)
    try:
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None


def get_global_properties(atoms):
    """
    Calculates global physical properties of the crystal structure.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        dict: Dictionary containing 'volume' and 'density'.
    """
    if atoms is None:
        return {"volume": np.nan, "density": np.nan}

    # Volume in Angstrom^3
    vol = atoms.get_volume()

    # Mass in atomic mass units (dalton)
    total_mass = sum(atoms.get_masses())

    # Density (amu / Angstrom^3)
    # Note: 1 amu/A^3 approx 1.66 g/cm^3. We keep it in amu/A^3 for consistency.
    density = total_mass / vol if vol > 1e-9 else np.nan

    return {"volume": vol, "density": density, "num_atoms": len(atoms)}


def get_chemically_resolved_bond_stats(atoms, cutoff=NEIGHBOR_CUTOFF):
    """
    Calculates statistics (mean, variance) of bond lengths for specific
    cation-anion pairs (Al-O, Ga-O, In-O).

    Args:
        atoms (ase.Atoms): The atomic structure.
        cutoff (float): Cutoff radius for neighbor search.

    Returns:
        dict: Features for each bond type (mean and variance).
    """
    if atoms is None:
        return {
            "mean_bond_Al-O": np.nan,
            "var_bond_Al-O": np.nan,
            "mean_bond_Ga-O": np.nan,
            "var_bond_Ga-O": np.nan,
            "mean_bond_In-O": np.nan,
            "var_bond_In-O": np.nan,
        }

    # Get neighbor list: i (center), j (neighbor), d (distance)
    # We use self_interaction=False to avoid counting atom with itself
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, cutoff)

    symbols = atoms.get_chemical_symbols()

    # Containers for bond distances
    bonds = {"Al-O": [], "Ga-O": [], "In-O": []}

    # Identify pairs
    # We iterate through the neighbor list.
    # Since neighbor_list returns both (i,j) and (j,i), we will process all.
    # We only care about Cation-O bonds.

    for k in range(len(i_indices)):
        idx_i = i_indices[k]
        idx_j = j_indices[k]
        dist = distances[k]

        sym_i = symbols[idx_i]
        sym_j = symbols[idx_j]

        # Check for Al-O, Ga-O, In-O (order doesn't matter for bond type)
        pair = None
        if (sym_i == "Al" and sym_j == "O") or (sym_i == "O" and sym_j == "Al"):
            pair = "Al-O"
        elif (sym_i == "Ga" and sym_j == "O") or (sym_i == "O" and sym_j == "Ga"):
            pair = "Ga-O"
        elif (sym_i == "In" and sym_j == "O") or (sym_i == "O" and sym_j == "In"):
            pair = "In-O"

        if pair:
            bonds[pair].append(dist)

    # Compute stats
    features = {}
    for pair_type in ["Al-O", "Ga-O", "In-O"]:
        dists = np.array(bonds[pair_type])
        if len(dists) > 0:
            features[f"mean_bond_{pair_type}"] = np.mean(dists)
            features[f"var_bond_{pair_type}"] = np.var(dists)
        else:
            # If no bonds of this type exist (e.g. no In atoms), use NaN
            features[f"mean_bond_{pair_type}"] = np.nan
            features[f"var_bond_{pair_type}"] = np.nan

    return features


def extract_geometry_features(
    metadata_df, load_cached_data=True, cache_name="geometry_features"
):
    """
    Main pipeline function to extract geometry-based features for a dataset.
    Implements caching to parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path' and 'id'.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_name (str): Identifier for the cache file.

    Returns:
        pd.DataFrame: DataFrame with extracted features, indexed by original index.
    """
    # Define cache path
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometry features from {cache_path}...")
        try:
            cached_df = pd.read_parquet(cache_path)
            # Ensure the cached data matches the requested indices
            # We align by index or id. Assuming metadata_df index is preserved.
            if len(cached_df) == len(metadata_df) and np.all(
                cached_df.index == metadata_df.index
            ):
                return cached_df
            else:
                print("Cached data index mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Extracting geometry features...")

    features_list = []

    for idx, row in metadata_df.iterrows():
        rel_path = row["file_path"]
        atoms = read_structure(rel_path)

        # Global properties
        global_props = get_global_properties(atoms)

        # Chemically resolved bond stats
        bond_stats = get_chemically_resolved_bond_stats(atoms, cutoff=NEIGHBOR_CUTOFF)

        # Combine
        combined = {**global_props, **bond_stats}
        features_list.append(combined)

    # Create DataFrame
    features_df = pd.DataFrame(features_list, index=metadata_df.index)

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        features_df.to_parquet(cache_path)
        print(f"Saved geometry features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return features_df
