import os
import random
import numpy as np
import pandas as pd
import ase.io
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    process_dataset,
)

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata containing IDs, targets (for train/val),
                      and file paths.
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


def read_structure(rel_path: str) -> ase.Atoms:
    """
    Reads an atomic structure from a file path relative to the input directory.

    Args:
        rel_path (str): Relative path to the .xyz file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure object.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Structure file not found at {full_path}")

    # Use ASE to read the XYZ file
    return ase.io.read(full_path)


def load_features(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads processed features for a given split. Uses the caching mechanism
    provided by the library configuration.

    This function loads the metadata, computes (or loads) the electro-geometric
    features using `process_dataset`, and merges them into a single DataFrame.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False, re-computes features.

    Returns:
        pd.DataFrame: DataFrame containing merged metadata and computed features.
    """
    # 1. Load Metadata
    metadata_df = load_metadata(split)

    # 2. Process or Load Features
    # process_dataset handles the caching logic internally (saving/loading from WORKING_DIR)
    features_df = process_dataset(metadata_df, split, load_cached_data=load_cached_data)

    # 3. Merge
    # We merge on 'id' to combine the targets (from metadata) with the computed features.
    # This results in a comprehensive dataset ready for training or inference.
    merged_df = pd.merge(metadata_df, features_df, on="id", how="inner")

    return merged_df


def load_sample_submission() -> pd.DataFrame:
    """
    Loads the sample submission file to check format or IDs.

    Returns:
        pd.DataFrame: The sample submission dataframe.
    """
    path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(path):
        # Fallback for case sensitivity
        path_alt = os.path.join(INPUT_DIR, "sampleSubmission.csv")
        if os.path.exists(path_alt):
            path = path_alt
        else:
            raise FileNotFoundError("Sample submission file not found.")

    return pd.read_csv(path)
