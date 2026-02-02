import os
import numpy as np
import pandas as pd
from library.config import Config

# Atomic masses (amu) for the elements present in the dataset
ATOMIC_MASSES = {
    "Al": 26.9815385,
    "Ga": 69.723,
    "In": 114.818,
    "O": 15.999,
}


def read_xyz_file(file_path):
    """
    Parses the custom XYZ format provided in the dataset.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        tuple: (lattice_vectors (np.ndarray), atoms (list of dicts))
    """
    lattice_vectors = []
    atoms = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "lattice_vector":
                    # Format: lattice_vector x y z
                    vec = [float(x) for x in parts[1:4]]
                    lattice_vectors.append(vec)
                elif parts[0] == "atom":
                    # Format: atom x y z Element
                    coords = [float(x) for x in parts[1:4]]
                    element = parts[4]
                    atoms.append({"element": element, "coords": coords})
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.zeros((3, 3)), []

    return np.array(lattice_vectors), atoms


def calculate_structural_properties(lattice_vectors, atoms):
    """
    Calculates volume, density, and number of atoms from parsed geometry data.

    Args:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atoms (list): List of atom dictionaries.

    Returns:
        dict: Dictionary containing 'geo_volume', 'geo_density', 'geo_num_atoms'.
    """
    # 1. Volume
    # Scalar triple product: v1 . (v2 x v3)
    if lattice_vectors.shape == (3, 3):
        v1 = lattice_vectors[0]
        v2 = lattice_vectors[1]
        v3 = lattice_vectors[2]
        volume = np.abs(np.dot(v1, np.cross(v2, v3)))
    else:
        volume = np.nan

    # 2. Number of atoms
    num_atoms = len(atoms)

    # 3. Density
    # Sum of masses / Volume
    total_mass = 0.0
    for atom in atoms:
        el = atom["element"]
        total_mass += ATOMIC_MASSES.get(el, 0.0)

    density = total_mass / volume if (volume > 0 and not np.isnan(volume)) else np.nan

    return {"geo_volume": volume, "geo_density": density, "geo_num_atoms": num_atoms}


def extract_geometry_features(metadata_df, cache_file_path, load_cached_data=True):
    """
    Extracts geometry features for a given metadata DataFrame.
    Handles caching using parquet files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        cache_file_path (str): Path to save/load the parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Original DataFrame merged with extracted geometry features.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file_path):
        print(f"Loading geometry features from cache: {cache_file_path}")
        try:
            cached_df = pd.read_parquet(cache_file_path)
            # Ensure required columns are present
            if (
                set(Config.GEO_COLS).issubset(cached_df.columns)
                and Config.ID_COL in cached_df.columns
            ):
                # Merge on ID to ensure alignment
                merged_df = pd.merge(
                    metadata_df, cached_df, on=Config.ID_COL, how="left"
                )
                return merged_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing geometry features for {len(metadata_df)} samples...")

    features_list = []

    for idx, row in metadata_df.iterrows():
        id_val = row[Config.ID_COL]
        rel_path = row[Config.FILE_PATH_COL]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            l_vecs, atoms = read_xyz_file(full_path)
            props = calculate_structural_properties(l_vecs, atoms)
        else:
            # Handle missing file case
            props = {
                "geo_volume": np.nan,
                "geo_density": np.nan,
                "geo_num_atoms": np.nan,
            }

        props[Config.ID_COL] = id_val
        features_list.append(props)

    features_df = pd.DataFrame(features_list)

    # 3. Save to cache
    # Save only ID and the new geometry columns
    cols_to_save = [Config.ID_COL] + Config.GEO_COLS
    try:
        features_df[cols_to_save].to_parquet(cache_file_path, index=False)
        print(f"Saved geometry features to cache: {cache_file_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file_path}: {e}")

    # Merge back to original dataframe
    merged_df = pd.merge(metadata_df, features_df, on=Config.ID_COL, how="left")

    return merged_df
