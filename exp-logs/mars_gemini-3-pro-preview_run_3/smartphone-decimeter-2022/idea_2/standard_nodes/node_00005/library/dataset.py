import os
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    AGG_FEATURES,
    TARGETS,
)
from library.features import generate_dataset


def create_dataset(split="train", load_cached_data=True):
    """
    Constructs the dataset for a given split (train, val, test).

    This function handles the high-level orchestration of data loading.
    It determines the appropriate metadata file based on the split and
    delegates the trip-wise processing, aggregation, concatenation, and
    caching to the library.features module.

    Args:
        split (str): The dataset split to generate. Must be 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load the dataset from the
                                 cache directory defined in config. If False or
                                 if the cache is missing, processes data from scratch.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated features, metadata,
                      and (for train/val) target residuals.
    """
    if split == "train":
        metadata_path = TRAIN_METADATA_PATH
    elif split == "val":
        metadata_path = VAL_METADATA_PATH
    elif split == "test":
        metadata_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

    # Delegate to the provided feature engineering library which implements
    # the iteration over metadata, processing of individual trips,
    # calculation of residuals (targets), and the required caching logic.
    df = generate_dataset(
        metadata_path=metadata_path, load_cached_data=load_cached_data, split_name=split
    )

    return df


def get_features_and_targets(split="train", load_cached_data=True):
    """
    Prepares the feature matrix X and target variable y for modeling.

    Args:
        split (str): The dataset split to prepare.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (X, y, df)
            X (pd.DataFrame): The feature matrix containing aggregated GNSS/IMU stats.
            y (pd.DataFrame or None): The target residuals (lat_error, lon_error).
                                      Returns None for the 'test' split.
            df (pd.DataFrame): The full dataframe, including metadata (tripId, timestamps)
                               and baseline WLS positions, useful for final reconstruction.
    """
    # Load the full dataset
    df = create_dataset(split, load_cached_data)

    # Filter for the defined aggregated features
    # We check intersection to avoid errors if a feature wasn't generated
    valid_features = [f for f in AGG_FEATURES if f in df.columns]
    X = df[valid_features]

    y = None

    # Extract targets only for training and validation splits
    # Test split will have placeholder targets which should be ignored
    if split in ["train", "val"]:
        if all(t in df.columns for t in TARGETS):
            y = df[TARGETS]
        else:
            # This case implies an issue with data generation (e.g. missing GT)
            print(f"Warning: Targets {TARGETS} not found in {split} dataset columns.")

    return X, y, df
