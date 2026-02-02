import os
import pandas as pd
import ase.io
from library.config import Config
from library.features import process_dataset


def load_geometry(file_path):
    """
    Parses an .xyz file using the ASE library.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        ase.Atoms: The atomic structure object, or None if reading fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Geometry file not found at {file_path}")

    try:
        atoms = ase.io.read(file_path)
        return atoms
    except Exception as e:
        print(f"Error reading geometry file {file_path}: {e}")
        return None


def build_dataset(split, load_cached_data=True, debug=False):
    """
    Constructs the dataset for a given split (train, val, test).
    Iterates through metadata, loads geometries, calls feature extraction functions,
    and assembles the final feature matrix.

    Implements caching mechanism:
    - Checks if parquet cache exists in Config.WORKING_DIR.
    - If load_cached_data is True and cache exists, loads it.
    - Otherwise, computes features from scratch and saves to cache.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes only a small subset of data for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features and targets (if available).
    """
    # Map split to specific metadata and cache file paths defined in Config
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_file = Config.TRAIN_FEATURES_FILE
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_file = Config.VAL_FEATURES_FILE
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
        cache_file = Config.TEST_FEATURES_FILE
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Use the process_dataset function from library.features
    # This function handles the iteration over metadata, loading of geometries via ase,
    # computation of Physical, RDF, and LEM features, and caching logic.
    df = process_dataset(
        metadata_path=metadata_path,
        cache_file=cache_file,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    return df


def get_feature_target_split(df, target_cols=None, drop_cols=None):
    """
    Splits the processed DataFrame into feature matrix X and target vector(s) y.

    Args:
        df (pd.DataFrame): The dataframe returned by build_dataset.
        target_cols (list, optional): List of target column names. Defaults to
                                      ['target_formation', 'target_bandgap'].
        drop_cols (list, optional): List of columns to exclude from features (e.g. 'id').
                                    Defaults to ['id'].

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a DataFrame of targets
               (or None if targets are not present).
    """
    if target_cols is None:
        target_cols = ["target_formation", "target_bandgap"]

    if drop_cols is None:
        drop_cols = ["id"]

    # Identify columns that are present in the dataframe
    present_targets = [col for col in target_cols if col in df.columns]

    # Define feature columns: all columns except targets and those explicitly dropped
    feature_cols = [
        col for col in df.columns if col not in target_cols and col not in drop_cols
    ]

    X = df[feature_cols]

    y = None
    if present_targets:
        y = df[present_targets]

    return X, y
