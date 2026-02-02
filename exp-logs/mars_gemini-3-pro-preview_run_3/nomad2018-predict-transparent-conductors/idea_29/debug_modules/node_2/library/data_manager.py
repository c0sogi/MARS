import os
import pandas as pd
import ase.io
import library.config as config
from library.feature_extraction import process_dataset


def load_metadata(dataset_type="train"):
    """
    Loads the metadata CSV file for the specified dataset type.

    Args:
        dataset_type (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if dataset_type == "train":
        path = config.TRAIN_METADATA_PATH
    elif dataset_type == "val":
        path = config.VAL_METADATA_PATH
    elif dataset_type == "test":
        path = config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. Must be 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    return pd.read_csv(path)


def load_geometry(file_path):
    """
    Loads the atomic geometry from an .xyz file.

    Args:
        file_path (str): Relative path to the .xyz file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atomic structure object.
    """
    full_path = os.path.join(config.INPUT_DIR, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    return ase.io.read(full_path)


def build_dataset(dataset_type="train", load_cached_data=True):
    """
    Constructs the full dataset with features for the specified type.

    This function orchestrates:
    1. Identification of the correct metadata file.
    2. Feature extraction (delegated to library.feature_extraction.process_dataset),
       which handles caching via parquet files.
    3. Merging of features with metadata (handled by process_dataset).
    4. Imputation of missing values (NaNs filled with 0.0 for features).

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        pd.DataFrame: The complete dataset ready for modeling.
    """
    # 1. Determine the metadata path
    if dataset_type == "train":
        meta_path = config.TRAIN_METADATA_PATH
    elif dataset_type == "val":
        meta_path = config.VAL_METADATA_PATH
    elif dataset_type == "test":
        meta_path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # 2. Process dataset (Extract features + Cache handling)
    # process_dataset returns a dataframe with metadata and extracted features merged.
    df = process_dataset(meta_path, load_cached_data=load_cached_data)

    # 3. Handle missing value imputation
    # We fill missing values in numeric feature columns with 0.0.
    # This is appropriate for histograms/RDFs where absence implies a count of 0.
    # We strictly avoid filling Target columns or IDs.

    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Columns that should NOT be filled if missing (Targets might be missing in test, ID is key)
    # Note: In train/val, targets should not be missing anyway.
    protected_cols = set(config.TARGET_COLS) | {"id"}

    cols_to_fill = [c for c in numeric_cols if c not in protected_cols]

    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].fillna(0.0)

    return df
