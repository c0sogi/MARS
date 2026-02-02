import os
import ase.io
import pandas as pd
from library.config import Config
from library.utils import load_metadata as _load_metadata_utils


def read_geometry(rel_path):
    """
    Reads a geometry file from the input directory and returns an ASE Atoms object.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    # ASE read returns an Atoms object containing Atom objects
    atoms = ase.io.read(full_path)
    return atoms


def load_metadata(split="train", limit=None):
    """
    Loads the metadata for a specific split (train, val, test).
    Wraps the utility function to add dataset limiting capabilities.

    Args:
        split (str): One of 'train', 'val', 'test'.
        limit (int, optional): Maximum number of rows to load. Useful for debugging.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    # Load full metadata using the provided utility
    df = _load_metadata_utils(split)

    # Apply limit if specified
    if limit is not None:
        # print(f"Limiting {split} dataset to {limit} samples.")
        df = df.iloc[:limit].copy()

    return df


def get_train_data(limit=None):
    """Convenience wrapper to load training data."""
    return load_metadata("train", limit=limit)


def get_val_data(limit=None):
    """Convenience wrapper to load validation data."""
    return load_metadata("val", limit=limit)


def get_test_data(limit=None):
    """Convenience wrapper to load test data."""
    return load_metadata("test", limit=limit)
