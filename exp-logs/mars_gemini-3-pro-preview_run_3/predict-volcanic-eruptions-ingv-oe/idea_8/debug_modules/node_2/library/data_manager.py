import os
import pandas as pd
import numpy as np
from library.config import TRAIN_META_PATH, VAL_META_PATH, TEST_META_PATH, WORKING_DIR
from library.feature_engineering import generate_features


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the given split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
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


def build_dataset(split: str, load_cached_data: bool = True, debug_size: int = None):
    """
    Constructs the feature matrix X and target vector y (if available) for the specified split.
    Leverages the feature_engineering library for processing and caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.
        debug_size (int, optional): Number of samples to process for debugging.

    Returns:
        tuple:
            - If split is 'train' or 'val': (X, y, segment_ids)
            - If split is 'test': (X, segment_ids)
    """
    # Determine metadata path and cache output name
    if split == "train":
        meta_path = TRAIN_META_PATH
        cache_name = "train_features"
    elif split == "val":
        meta_path = VAL_META_PATH
        cache_name = "val_features"
    elif split == "test":
        meta_path = TEST_META_PATH
        cache_name = "test_features"
    else:
        raise ValueError(f"Invalid split: {split}")

    # Append debug suffix to cache name to prevent overwriting full features with debug features
    if debug_size is not None:
        cache_name = f"{cache_name}_debug_{debug_size}"

    # Generate features (handles caching internally)
    df = generate_features(
        metadata_path=meta_path,
        output_name=cache_name,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    if df.empty:
        raise ValueError("Generated dataset is empty.")

    # Extract components
    segment_ids = df["segment_id"]

    # Identify feature columns (drop metadata/target columns)
    # Note: 'time_to_eruption' is added by generate_features (as NaN for test)
    drop_cols = ["segment_id"]
    if "time_to_eruption" in df.columns:
        drop_cols.append("time_to_eruption")

    X = df.drop(columns=drop_cols)

    if split in ["train", "val"]:
        y = df["time_to_eruption"]
        return X, y, segment_ids
    else:
        # For test, we generally don't return y (it's all NaN anyway)
        return X, segment_ids
