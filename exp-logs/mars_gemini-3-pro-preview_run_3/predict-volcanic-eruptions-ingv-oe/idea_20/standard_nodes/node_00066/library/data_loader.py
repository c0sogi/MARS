import os
import pandas as pd
import numpy as np
from library.config import Config
from library.feature_engineering import generate_features


def load_metadata(path):
    """
    Loads the metadata CSV file.

    Args:
        path (str): Path to the metadata CSV file.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path)


def create_dataset(mode, load_cached_data=True, debug_limit=None):
    """
    Generates or loads the dataset for the specified mode.

    This function acts as a wrapper around the feature engineering pipeline.
    It handles path resolution, invokes the parallel feature extraction (with caching),
    and formats the output into Numpy arrays suitable for model training/inference.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load features from Parquet cache.
                                 If False or cache missing, computes features from scratch.
        debug_limit (int, optional): If provided, limits the number of samples processed
                                     (useful for debugging).

    Returns:
        tuple:
            - If mode is 'train' or 'val': (X, y)
                X (np.ndarray): Feature matrix.
                y (np.ndarray): Target vector (time_to_eruption).
            - If mode is 'test': (X, segment_ids)
                X (np.ndarray): Feature matrix.
                segment_ids (np.ndarray): Array of segment IDs corresponding to X.
    """
    # 1. Resolve Paths based on Mode
    if mode == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        feat_path = Config.TRAIN_FEATURES_PATH
    elif mode == "val":
        meta_path = Config.VAL_METADATA_PATH
        feat_path = Config.VAL_FEATURES_PATH
    elif mode == "test":
        meta_path = Config.TEST_METADATA_PATH
        feat_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'train', 'val', or 'test'.")

    # 2. Generate or Load Features
    # The generate_features function from the library handles:
    # - Reading metadata
    # - Parallel processing of sensor files
    # - Caching results to Parquet
    # - Loading from cache if requested and available
    df = generate_features(
        metadata_path=meta_path,
        output_path=feat_path,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Format Output
    # Define columns to exclude from the feature matrix X
    non_feature_cols = ["segment_id", "time_to_eruption"]

    # Extract Feature Matrix X
    # Drop known non-feature columns if they exist
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].values

    if mode in ["train", "val"]:
        # Extract Target y
        if "time_to_eruption" not in df.columns:
            raise KeyError(
                f"Target column 'time_to_eruption' missing in {mode} dataset."
            )
        y = df["time_to_eruption"].values
        return X, y

    elif mode == "test":
        # Extract Segment IDs for submission mapping
        if "segment_id" not in df.columns:
            raise KeyError(f"Column 'segment_id' missing in {mode} dataset.")
        segment_ids = df["segment_id"].values
        return X, segment_ids
