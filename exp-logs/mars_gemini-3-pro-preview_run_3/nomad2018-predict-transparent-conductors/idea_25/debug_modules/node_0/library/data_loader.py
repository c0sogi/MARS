import os
import pandas as pd
import ase.io
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    RANDOM_SEED,
)


def load_metadata(split: str = "train", max_rows: int = None) -> pd.DataFrame:
    """
    Loads the metadata DataFrame for a specific data split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        max_rows (int, optional): Maximum number of rows to load. Useful for debugging/testing.

    Returns:
        pd.DataFrame: The loaded metadata containing IDs, targets (if applicable), and file paths.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    # Load data
    df = pd.read_csv(path)

    # Sample if max_rows is specified
    if max_rows is not None and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Loaded subset of {split} data: {len(df)} rows.")

    return df


def read_geometry(rel_path: str) -> ase.Atoms:
    """
    Reads an atomic geometry file (.xyz) and returns an ASE Atoms object.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').
                        This path is relative to the INPUT_DIR.

    Returns:
        ase.Atoms: The atomic structure object.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    try:
        # ASE's read function automatically detects format or uses the extension
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        raise RuntimeError(f"Error reading geometry file {full_path}: {e}")
