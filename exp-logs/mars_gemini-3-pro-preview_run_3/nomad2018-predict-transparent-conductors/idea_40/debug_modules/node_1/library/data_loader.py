import os
import pandas as pd
import ase.io
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
)


def load_structure(relative_path: str) -> ase.Atoms:
    """
    Loads the atomic structure from an .xyz file.

    Args:
        relative_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: An ASE Atoms object containing the atomic coordinates and lattice information.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full_path = os.path.join(INPUT_DIR, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    try:
        # Cite debug_lesson_3: Explicitly Specify File Formats When Extensions Are Misleading
        atoms = ase.io.read(full_path, format="aims")
        return atoms
    except Exception as e:
        raise RuntimeError(f"Failed to parse geometry file {full_path}: {str(e)}")


def load_metadata(
    split: str, debug: bool = DEBUG_MODE, sample_size: int = DEBUG_SAMPLE_SIZE
) -> pd.DataFrame:
    """
    Loads the metadata CSV file for a specific data split.

    Args:
        split (str): The data split to load. Options: 'train', 'val', 'test'.
        debug (bool): If True, returns a subsample of the data for debugging.
        sample_size (int): The number of rows to return if debug is True.

    Returns:
        pd.DataFrame: DataFrame containing the metadata for the requested split.

    Raises:
        ValueError: If an invalid split name is provided.
        FileNotFoundError: If the metadata file does not exist.
    """
    split = split.lower()
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be one of 'train', 'val', 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file for split '{split}' not found at {path}"
        )

    df = pd.read_csv(path)

    if debug:
        # Use a fixed random state for reproducibility in debug mode
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Debug mode enabled: Loaded {len(df)} samples from {split} split.")

    return df
