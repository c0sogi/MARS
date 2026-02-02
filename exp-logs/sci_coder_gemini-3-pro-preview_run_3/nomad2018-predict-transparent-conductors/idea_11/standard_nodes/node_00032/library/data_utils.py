import os
import pandas as pd
import ase.io
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    RANDOM_SEED,
)


def load_metadata(split: str, sample_size: int = None) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', 'test'.
        sample_size (int, optional): If provided, returns a random sample of this size for debugging.

    Returns:
        pd.DataFrame: The metadata dataframe containing features, targets (if available), and file paths.
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

    df = pd.read_csv(path)

    if sample_size is not None and sample_size > 0:
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(
                drop=True
            )

    return df


def read_geometry(rel_path: str):
    """
    Reads an XYZ geometry file using ASE.

    Args:
        rel_path (str): Relative path to the geometry file from the input directory (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The parsed atomic structure object containing positions, cell, and atomic numbers.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at {full_path}")

    # Parse the xyz file
    # ASE reads xyz files and returns an Atoms object
    try:
        # Explicitly specify format='aims' as .xyz extension is misleading for these files
        atoms = ase.io.read(full_path, format="aims")
    except Exception as e:
        raise RuntimeError(f"Failed to parse geometry file at {full_path}: {str(e)}")

    return atoms
