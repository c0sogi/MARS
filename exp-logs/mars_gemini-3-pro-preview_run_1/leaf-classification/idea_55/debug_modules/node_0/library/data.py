import os
import pandas as pd
import numpy as np
from library import config, utils, features


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for the specified split, performing additive fusion of
    tabular and geometric features, and enforcing alphanumeric column ordering.

    This function leverages the library.features module to handle the extraction
    of integral-geometric descriptors and their fusion with the original tabular data.
    It then prepares the final feature matrix X, target vector y, and ID vector
    strictly in float64 precision.

    Args:
        split (str): The dataset split to load. Options: 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load pre-computed features
                                 from the cache directory defined in config.

    Returns:
        tuple: A tuple containing (X, y, ids).
            - X (pd.DataFrame): The feature matrix with float64 precision.
                                Columns are sorted alphanumerically.
            - y (pd.Series or None): The target labels (species). None for 'test' split.
            - ids (pd.Series): The unique image identifiers.
    """
    logger = utils.setup_logger()

    # Delegate to library.features to handle metadata loading, geometric extraction,
    # caching, and merging (Additive Fusion).
    if split == "train":
        df = features.get_train_data(load_cached_data=load_cached_data)
    elif split == "val":
        df = features.get_val_data(load_cached_data=load_cached_data)
    elif split == "test":
        df = features.get_test_data(load_cached_data=load_cached_data)
    else:
        raise ValueError(
            f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
        )

    # Respect debug sampling even if loading from a full cache
    if config.DEBUG_SAMPLE_SIZE is not None:
        logger.info(
            f"Debug mode: Subsetting {split} data to {config.DEBUG_SAMPLE_SIZE} rows."
        )
        df = df.head(config.DEBUG_SAMPLE_SIZE)

    # Define columns that are not features
    # Note: 'full_path' is not in metadata but added to exclude list for safety
    non_feature_cols = {"id", "species", "file_path", "full_path"}

    # Identify feature columns
    # The dataframe contains original tabular features + new geometric features
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Enforce Alphanumeric Column Ordering
    # This ensures a deterministic memory layout for the model
    feature_cols.sort()

    # Extract Feature Matrix X
    X = df[feature_cols].copy()

    # Enforce float64 precision (Sanitized Integral-Geometric High-Precision)
    X = utils.enforce_float64(X)

    # Extract Target y and IDs
    ids = df["id"].copy()

    if "species" in df.columns:
        y = df["species"].copy()
    else:
        y = None

    logger.info(
        f"[{split.upper()}] Loaded {len(X)} samples with {len(feature_cols)} features."
    )

    return X, y, ids
