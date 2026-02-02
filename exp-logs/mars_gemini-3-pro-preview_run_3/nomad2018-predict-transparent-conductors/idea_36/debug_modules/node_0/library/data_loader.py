import os
import pandas as pd
import ase.io
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    RANDOM_SEED,
)


def load_metadata(split: str, debug: bool = DEBUG) -> pd.DataFrame:
    """
    Loads the metadata for a specific dataset split.

    Args:
        split (str): The dataset split to load. Options are 'train', 'val', or 'test'.
        debug (bool): If True, returns a small subsample of the data for debugging purposes.
                      Defaults to the global DEBUG setting in config.

    Returns:
        pd.DataFrame: DataFrame containing metadata and targets (if available).
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
        raise FileNotFoundError(
            f"Metadata file not found at {path}. Please ensure metadata generation was successful."
        )

    df = pd.read_csv(path)

    if debug:
        sample_n = min(len(df), DEBUG_SAMPLE_SIZE)
        df = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[DEBUG] Loaded {len(df)} samples from {split} metadata.")

    return df


def read_geometry(rel_path: str) -> ase.Atoms:
    """
    Reads a geometry file from the input directory and returns an ASE Atoms object.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz')
                        as found in the metadata 'file_path' column.

    Returns:
        ase.Atoms: The atomic structure object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If parsing fails.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    try:
        # ASE handles .xyz format automatically
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse geometry file at {full_path}. Error: {str(e)}"
        )
