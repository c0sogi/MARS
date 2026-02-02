import os
import pandas as pd
import ase.io
from library.config import Config


def load_metadata(split: str, sample_size: int = None) -> pd.DataFrame:
    """
    Loads the metadata DataFrame for a given split.

    Args:
        split (str): The dataset split to load. Must be one of 'train', 'val', or 'test'.
        sample_size (int, optional): If provided, limits the number of rows loaded.
                                     If None, checks Config.DEBUG to determine if subsampling is needed.

    Returns:
        pd.DataFrame: The requested metadata containing IDs, targets (if available), and file paths.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # Handle dataset limiting for debugging or quick iteration
    if sample_size is not None:
        df = df.iloc[:sample_size]
    elif Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    return df


def read_geometry(rel_path: str) -> ase.Atoms:
    """
    Reads a geometry file from the input directory and returns an ASE Atoms object.

    Args:
        rel_path (str): The relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If parsing fails.
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    try:
        # ASE handles the .xyz format parsing
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        raise RuntimeError(f"Failed to read geometry from {full_path}: {e}")


def get_geometry_path(rel_path: str) -> str:
    """
    Resolves the full system path for a given relative geometry path.

    Args:
        rel_path (str): Relative path (e.g., 'test/10/geometry.xyz').

    Returns:
        str: Absolute or relative path from the current working directory.
    """
    return os.path.join(Config.INPUT_DIR, rel_path)
