import os
import cv2
import numpy as np
import pandas as pd
from library import config


def get_augmented_dataset(split_name, load_cached_data=True):
    """
    Loads the dataset for a specific split.
    Note: Augmentation is removed to avoid heteroscedasticity (Cite solution_lesson_node_00028).
    We keep the function name for compatibility with dataset.py, but it no longer augments.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The dataset.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Determine source path based on split name
    if split_name == "train":
        source_path = config.TRAIN_DATA_PATH
    elif split_name == "val":
        source_path = config.VAL_DATA_PATH
    elif split_name == "test":
        source_path = config.TEST_DATA_PATH
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    print(f"Loading {split_name} data from {source_path}...")

    # Load metadata
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    return df
