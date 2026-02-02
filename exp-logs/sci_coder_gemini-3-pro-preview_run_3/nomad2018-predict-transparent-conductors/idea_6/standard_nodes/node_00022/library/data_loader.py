import os
import pandas as pd
import ase.io
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
)


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe containing material IDs, features, and file paths.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def load_geometry(rel_path: str):
    """
    Loads the geometry from a file using ASE.

    Note: The dataset files have an .xyz extension but the content follows the
    FHI-aims format (lattice_vector and atom keywords). We explicitly specify
    format='aims' to handle this.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: Atoms object representing the crystal structure, or None if loading fails.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        print(f"Warning: Geometry file not found at {full_path}")
        return None

    try:
        # Force 'aims' format as the content matches FHI-aims structure despite .xyz extension
        atoms = ase.io.read(full_path, format="aims")
        return atoms
    except Exception as e:
        # Fallback to auto-detection if specific format fails
        try:
            atoms = ase.io.read(full_path)
            return atoms
        except Exception as e2:
            print(f"Error reading geometry file {full_path}: {e}")
            return None


def get_dataset(split: str, limit: int = None):
    """
    Generator that yields metadata and geometry for each sample in the split.

    This function allows iterating over the dataset without loading all geometry
    files into memory at once.

    Args:
        split (str): 'train', 'val', or 'test'.
        limit (int, optional): Maximum number of samples to yield. Useful for debugging.

    Yields:
        tuple: (pd.Series, ase.Atoms)
            - row: Metadata row for the sample.
            - atoms: ASE Atoms object containing geometry.
    """
    df = load_metadata(split)

    if limit is not None:
        df = df.head(limit)

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        atoms = load_geometry(rel_path)

        if atoms is not None:
            yield row, atoms
