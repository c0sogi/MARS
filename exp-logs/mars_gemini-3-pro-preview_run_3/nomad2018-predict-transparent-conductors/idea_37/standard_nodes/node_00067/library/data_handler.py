import os
import pandas as pd
import numpy as np
import ase.io
from ase import Atoms
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_37"

# Set random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_metadata(
    split: str, limit: int = None, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Loads the metadata CSV for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.
        limit (int, optional): If set, return only the first N rows. Useful for debugging.
        random_state (int): Seed for reproducibility if sampling (currently just head is used).

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    file_name = f"{split}_metadata.csv"
    file_path = os.path.join(METADATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    df = pd.read_csv(file_path)

    if limit is not None:
        df = df.head(limit)
        print(f"Loaded {len(df)} rows from {split} metadata (limited).")
    else:
        print(f"Loaded {len(df)} rows from {split} metadata.")

    return df


def get_file_paths(df: pd.DataFrame) -> pd.Series:
    """
    Resolves the full file paths for the geometry files listed in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing a 'file_path' column (relative path).

    Returns:
        pd.Series: Series containing full file paths.
    """
    if "file_path" not in df.columns:
        raise ValueError("Dataframe must contain 'file_path' column.")

    return df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))


def load_geometry(
    df: pd.DataFrame, load_cached_data: bool = True, cache_name: str = None
) -> list:
    """
    Parses .xyz files corresponding to the dataframe rows into ASE Atoms objects.

    Args:
        df (pd.DataFrame): Metadata dataframe with 'file_path' column.
        load_cached_data (bool): Flag to indicate if caching should be used.
                                 (Note: Direct object caching is skipped to avoid pickle,
                                 reading from source is fast and robust).
        cache_name (str): Identifier for the cache (unused here but kept for interface consistency).

    Returns:
        list[Atoms]: A list of ASE Atoms objects corresponding to the rows in df.
    """
    # Ensure cache directory exists if we were to use it
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Note: We choose not to cache ASE objects using pickle to strictly follow the
    # "no pickle" constraint. Serializing Atoms to npz is complex and reading
    # ~2000 small text files is sufficiently fast.

    atoms_list = []
    full_paths = get_file_paths(df)

    print(f"Loading geometry for {len(df)} samples...")

    for path in full_paths:
        if not os.path.exists(path):
            # In case of missing file, we might append None or raise error.
            # Raising error is safer for data integrity.
            raise FileNotFoundError(f"Geometry file not found: {path}")

        try:
            # ase.io.read returns an Atoms object
            atoms = ase.io.read(path)
            atoms_list.append(atoms)
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            raise e

    print(f"Successfully loaded {len(atoms_list)} geometry objects.")
    return atoms_list
