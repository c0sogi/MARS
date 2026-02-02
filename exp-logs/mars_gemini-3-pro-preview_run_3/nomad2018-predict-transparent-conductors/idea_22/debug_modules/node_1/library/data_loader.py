import os
import pandas as pd
import ase.io
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)


def load_metadata(split="train", limit=None):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        limit (int, optional): If provided, limits the number of rows loaded.

    Returns:
        pd.DataFrame: The metadata dataframe.
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
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    df = pd.read_csv(path)

    if limit is not None:
        df = df.head(limit)

    return df


def load_geometry(rel_path):
    """
    Loads a single geometry file from a relative path.

    Args:
        rel_path (str): Relative path to the .xyz file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure object.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    # ase.io.read parses the xyz file into an Atoms object
    # Cite debug_lesson_3: Explicitly Specify File Formats When Extensions Are Misleading
    atoms = ase.io.read(full_path, format="aims")
    return atoms


def load_geometries(df):
    """
    Loads geometry objects for all entries in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing a 'file_path' column.

    Returns:
        list[ase.Atoms]: A list of ase.Atoms objects corresponding to the rows in df.
    """
    if "file_path" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'file_path' column to load geometries."
        )

    geometries = []
    for path in df["file_path"]:
        geometries.append(load_geometry(path))

    return geometries
