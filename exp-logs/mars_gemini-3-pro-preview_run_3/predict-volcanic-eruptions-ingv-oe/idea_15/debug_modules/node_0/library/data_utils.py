import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
)


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata for a specific data split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe containing segment_ids and file_paths.
    """
    if split == "train":
        path = TRAIN_META_PATH
    elif split == "val":
        path = VAL_META_PATH
    elif split == "test":
        path = TEST_META_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def load_sensor_segment(file_path: str) -> pd.DataFrame:
    """
    Loads a specific sensor data segment from a CSV file.

    Args:
        file_path (str): The relative file path to the sensor csv (e.g., 'train/123.csv').
                         This is typically obtained from the metadata dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the sensor readings.
                      Loaded as float32 to handle NaNs and optimize memory.
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Sensor file not found at {full_path}")

    # Load as float32 as per dataset description to handle NaNs and reduce memory footprint
    return pd.read_csv(full_path, dtype="float32")


def save_features(df: pd.DataFrame, path: str) -> None:
    """
    Saves a dataframe to a Parquet file. Used for caching processed features.

    Args:
        df (pd.DataFrame): The dataframe to save.
        path (str): The destination path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_features(path: str) -> pd.DataFrame:
    """
    Loads a dataframe from a Parquet file. Used for retrieving cached features.

    Args:
        path (str): The path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached feature file not found at {path}")

    return pd.read_parquet(path)
