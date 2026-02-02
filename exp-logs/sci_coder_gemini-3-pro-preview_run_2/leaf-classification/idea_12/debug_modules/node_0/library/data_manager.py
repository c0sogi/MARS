import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    FEATURE_GROUPS,
    NUM_FEATURES_PER_GROUP,
    META_FEATURES,
)
from library.image_features import get_augmented_dataset


def get_all_feature_names():
    """
    Generates the list of all feature column names, including the original
    extracted features and the new morphological meta-features.

    Returns:
        list: A list of column names strings.
    """
    feature_cols = []
    # Add original features: margin_1..64, shape_1..64, texture_1..64
    for group in FEATURE_GROUPS:
        for i in range(1, NUM_FEATURES_PER_GROUP + 1):
            feature_cols.append(f"{group}_{i}")

    # Add new meta-features
    feature_cols.extend(META_FEATURES)
    return feature_cols


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the augmented dataset for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from the parquet cache.

    Returns:
        tuple: (X, y, ids)
            - X (pd.DataFrame): Feature matrix.
            - y (pd.Series or None): Target labels (species). None for 'test' split.
            - ids (pd.Series): Image IDs.
    """
    # Map split to corresponding file paths
    if split == "train":
        metadata_path = TRAIN_CSV
        cache_path = CACHE_TRAIN_FEATURES
    elif split == "val":
        metadata_path = VAL_CSV
        cache_path = CACHE_VAL_FEATURES
    elif split == "test":
        metadata_path = TEST_CSV
        cache_path = CACHE_TEST_FEATURES
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Load augmented dataframe using the library function
    # This handles feature extraction and caching internally
    df = get_augmented_dataset(
        metadata_path, cache_path, load_cached_data=load_cached_data
    )

    # Identify feature columns
    feature_cols = get_all_feature_names()

    # Verify all expected columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in loaded dataset: {missing_cols}")

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
