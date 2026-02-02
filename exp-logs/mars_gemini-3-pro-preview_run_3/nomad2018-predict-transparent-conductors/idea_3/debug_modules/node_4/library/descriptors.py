import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_io import read_geometry
from library.utils import load_or_compute


def get_volume(atoms):
    """
    Calculates the volume of the unit cell in Angstrom^3.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        float: Volume of the unit cell.
    """
    # ASE atoms object has a get_volume method
    try:
        return atoms.get_volume()
    except Exception:
        return 0.0


def get_density(atoms):
    """
    Calculates the density of the material.
    Density is proportional to sum of atomic masses divided by volume.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        float: Density value.
    """
    try:
        vol = atoms.get_volume()
        if vol < 1e-9:
            return 0.0
        # get_masses returns an array of atomic masses
        total_mass = sum(atoms.get_masses())
        return total_mass / vol
    except Exception:
        return 0.0


def _compute_descriptors_internal(df):
    """
    Internal function to compute descriptors for a dataframe.
    This function is passed to load_or_compute.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'file_path'.

    Returns:
        pd.DataFrame: Dataframe with 'volume' and 'density' columns.
    """
    volumes = []
    densities = []

    # Iterate through the dataframe
    # We use the file_path column which contains relative paths like 'train/1/geometry.xyz'
    for _, row in df.iterrows():
        try:
            rel_path = row["file_path"]
            atoms = read_geometry(rel_path)

            vol = get_volume(atoms)
            dens = get_density(atoms)

            volumes.append(vol)
            densities.append(dens)
        except Exception as e:
            print(f"Error processing geometry for ID {row.get('id', 'unknown')}: {e}")
            volumes.append(0.0)
            densities.append(0.0)

    # Create result dataframe
    result_df = pd.DataFrame({"volume": volumes, "density": densities})

    return result_df


def compute_descriptors(df, cache_name="descriptors.parquet", load_cached_data=True):
    """
    Computes physical descriptors (volume, density) for the given metadata dataframe.
    Implements caching using parquet files.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'file_path'.
        cache_name (str): Name of the cache file (saved in Config.WORKING_DIR).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe containing the computed descriptors.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # Use the utility function to handle caching logic
    descriptors = load_or_compute(
        cache_path=cache_path,
        compute_func=_compute_descriptors_internal,
        load_cached_data=load_cached_data,
        df=df,
    )

    return descriptors
