import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TARGET_COLS,
    WORKING_DIR,
)
from library.features import process_dataset


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
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


def build_feature_matrix(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Constructs the feature matrix for a given dataset split.

    This function delegates the geometric processing to library.features.process_dataset,
    which handles reading XYZ files, computing descriptors (Global, RDF, Hierarchical Moments),
    merging with tabular metadata, and caching the result to a Parquet file.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from the cache first.
                                 If False or cache is missing, re-computes features.

    Returns:
        pd.DataFrame: A dataframe containing both tabular metadata and computed geometric features.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
        output_path = TRAIN_FEATURES_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
        output_path = VAL_FEATURES_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
        output_path = TEST_FEATURES_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Ensure the working directory exists (safety check)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # process_dataset implements the caching logic:
    # 1. Checks if load_cached_data is True and file exists.
    # 2. If not, computes features from scratch.
    # 3. Saves to output_path (Parquet).
    # 4. Returns the dataframe.
    df = process_dataset(meta_path, output_path, load_cached_data=load_cached_data)

    return df


def log_transform_targets(
    df: pd.DataFrame, targets: list = TARGET_COLS
) -> pd.DataFrame:
    """
    Applies log(1 + x) transformation to the target columns.

    Args:
        df (pd.DataFrame): Dataframe containing target columns.
        targets (list): List of column names to transform.

    Returns:
        pd.DataFrame: Dataframe with transformed targets.
    """
    df_transformed = df.copy()
    for col in targets:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def inverse_transform_targets(predictions: np.ndarray) -> np.ndarray:
    """
    Applies exp(x) - 1 transformation to reverse the log transformation.

    Args:
        predictions (np.ndarray): Array of log-transformed predictions.

    Returns:
        np.ndarray: Array of predictions in original scale.
    """
    return np.expm1(predictions)
