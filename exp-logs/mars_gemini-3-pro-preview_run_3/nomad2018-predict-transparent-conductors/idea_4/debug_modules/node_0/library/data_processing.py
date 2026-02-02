import os
import pandas as pd
import numpy as np
import ase.io
from library.config import Config


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def read_geometry(file_path):
    """
    Reads an XYZ file and returns an ASE Atoms object.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    if not os.path.exists(full_path):
        return None
    try:
        # The XYZ files contain custom lattice_vector headers which ASE's default 'xyz' reader
        # might handle if formatted specifically, but often robust reading requires care.
        # Given the description "Created using the Atomic Simulation Environment (ASE)",
        # ase.io.read should handle it correctly.
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None


def extract_physical_descriptors(df):
    """
    Extracts physical descriptors from geometry files and combines them with tabular data.
    Returns a DataFrame with features and a list of ASE Atoms objects.
    """
    volumes = []
    densities = []
    atoms_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        atoms = read_geometry(row[Config.FILE_PATH_COL])

        if atoms is None:
            continue

        # Calculate Volume from lattice parameters provided in metadata
        # This is often more robust than relying on the XYZ file if the cell isn't explicitly set in the Atoms object
        a = row["lattice_vector_1_ang"]
        b = row["lattice_vector_2_ang"]
        c = row["lattice_vector_3_ang"]
        alpha = np.radians(row["lattice_angle_alpha_degree"])
        beta = np.radians(row["lattice_angle_beta_degree"])
        gamma = np.radians(row["lattice_angle_gamma_degree"])

        # Volume of a parallelepiped
        vol = (
            a
            * b
            * c
            * np.sqrt(
                1
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
        )

        # Density (Atomic Mass Units / Angstrom^3)
        # Sum of atomic masses
        total_mass = sum(atoms.get_masses())
        density = total_mass / vol if vol > 1e-9 else 0.0

        volumes.append(vol)
        densities.append(density)
        atoms_list.append(atoms)
        valid_indices.append(idx)

    # Filter dataframe to valid rows and add new features
    df_processed = df.loc[valid_indices].copy()
    df_processed["volume"] = volumes
    df_processed["density"] = densities

    return df_processed, atoms_list


def process_data(split="train", load_cached_data=True):
    """
    Main function to load, process, and cache data.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        tuple: (pd.DataFrame, list[ase.Atoms])
    """
    # Determine cache path
    if split == "train":
        cache_path = Config.TRAIN_FEATS_PATH
    elif split == "val":
        cache_path = Config.VAL_FEATS_PATH
    elif split == "test":
        cache_path = Config.TEST_FEATS_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        df = pd.read_parquet(cache_path)

        # Re-load atoms objects (cannot be cached easily in parquet)
        # We iterate through the dataframe to ensure alignment
        atoms_list = []
        valid_indices = []

        # We need to re-index the dataframe to ensure iteration order matches
        df = df.reset_index(drop=True)

        for idx, row in df.iterrows():
            atoms = read_geometry(row[Config.FILE_PATH_COL])
            if atoms is not None:
                atoms_list.append(atoms)
                valid_indices.append(idx)

        # If some files are missing (unlikely if cache exists but good for safety)
        if len(valid_indices) != len(df):
            print(
                f"Warning: {len(df) - len(valid_indices)} geometry files could not be read. Filtering DataFrame."
            )
            df = df.loc[valid_indices].reset_index(drop=True)

        return df, atoms_list

    # If not cached or cache loading failed/disabled
    print(f"Processing {split} data from scratch...")

    # Load metadata
    df_meta = load_metadata(split)

    # Extract features
    df_features, atoms_list = extract_physical_descriptors(df_meta)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved features to {cache_path}")

    return df_features, atoms_list
