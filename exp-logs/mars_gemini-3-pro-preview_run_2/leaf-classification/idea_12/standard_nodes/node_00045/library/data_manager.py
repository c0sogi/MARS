import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    FEATURE_GROUPS,
    NUM_FEATURES_PER_GROUP,
)


def get_all_feature_names():
    """
    Generates the list of all feature column names.
    """
    feature_cols = []
    # Add original features: margin1..64, shape1..64, texture1..64
    for group in FEATURE_GROUPS:
        for i in range(1, NUM_FEATURES_PER_GROUP + 1):
            feature_cols.append(f"{group}{i}")
    return feature_cols


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for a specific split.
    """
    # Map split to corresponding file paths
    if split == "train":
        metadata_path = TRAIN_CSV
    elif split == "val":
        metadata_path = VAL_CSV
    elif split == "test":
        metadata_path = TEST_CSV
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Load dataframe directly
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Identify feature columns
    feature_cols = get_all_feature_names()

    # Extract components
    X = df[feature_cols].copy()
    ids = df["id"].copy()

    if "species" in df.columns:
        y = df["species"].copy()
    else:
        y = None

    return X, y, ids


def load_combined_train_val(load_cached_data=True):
    """
    Loads and concatenates the training and validation datasets.
    Useful for final model training to maximize data usage.

    Args:
        load_cached_data (bool): Whether to attempt loading from the parquet cache.

    Returns:
        tuple: (X_combined, y_combined, ids_combined)
    """
    print("Loading combined Train and Validation sets...")

    X_train, y_train, ids_train = load_dataset("train", load_cached_data)
    X_val, y_val, ids_val = load_dataset("val", load_cached_data)

    X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    ids_combined = pd.concat([ids_train, ids_val], axis=0, ignore_index=True)

    return X_combined, y_combined, ids_combined
