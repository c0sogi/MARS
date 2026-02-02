import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import CACHE_DIR, METADATA_DIR
from library.image_processing import process_all_images


def get_feature_groups(feature_names):
    """
    Identifies column indices for different feature groups based on column names.

    Args:
        feature_names (list): List of feature column names.

    Returns:
        dict: Mapping of group name to list of integer indices.
              Groups: 'margin', 'shape', 'texture', 'morphometrics', 'global'.
    """
    groups = {
        "margin": [],
        "shape": [],
        "texture": [],
        "morphometrics": [],
        "global": [],  # All features
    }

    for idx, name in enumerate(feature_names):
        groups["global"].append(idx)

        if "margin" in name:
            groups["margin"].append(idx)
        elif "shape" in name:
            groups["shape"].append(idx)
        elif "texture" in name:
            groups["texture"].append(idx)
        # Morphometric features from image_processing.py
        elif any(
            x in name
            for x in ["hu_", "aspect_ratio", "solidity", "extent", "eccentricity"]
        ):
            groups["morphometrics"].append(idx)

    return groups


def load_dataset(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Merges provided tabular features with extracted morphometric features.
    Handles caching of the merged datasets.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        dict: A dictionary containing:
            - 'X_train': np.ndarray (float64)
            - 'y_train': np.ndarray (int)
            - 'X_val': np.ndarray (float64)
            - 'y_val': np.ndarray (int)
            - 'X_test': np.ndarray (float64)
            - 'test_ids': np.ndarray (int/str)
            - 'classes': np.ndarray (str) - Class names corresponding to labels
            - 'feature_names': list (str) - Names of all feature columns
            - 'feature_groups': dict - Indices for feature subsets
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    train_cache = os.path.join(CACHE_DIR, "train_merged.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_merged.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_merged.parquet")

    # Check if we can load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading merged datasets from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
    else:
        print("Loading metadata and processing images...")
        # 1. Load Metadata
        meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        meta_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        # 2. Process Images (Extract Morphometrics)
        # This function handles its own caching of the extraction process
        morph_train = process_all_images(meta_train, load_cached_data=load_cached_data)
        morph_val = process_all_images(meta_val, load_cached_data=load_cached_data)
        morph_test = process_all_images(meta_test, load_cached_data=load_cached_data)

        # 3. Merge DataFrames
        # Inner join on 'id' to ensure alignment
        df_train = pd.merge(meta_train, morph_train, on="id", how="inner")
        df_val = pd.merge(meta_val, morph_val, on="id", how="inner")
        df_test = pd.merge(meta_test, morph_test, on="id", how="inner")

        # 4. Save to Cache
        print("Saving merged datasets to cache...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

    # 5. Prepare Features and Labels

    # Identify feature columns (exclude non-feature cols)
    exclude_cols = ["id", "species", "image_path"]
    feature_names = [c for c in df_train.columns if c not in exclude_cols]

    # Encode Labels
    # We fit the encoder on the union of train and val species to ensure consistency
    all_species = pd.concat([df_train["species"], df_val["species"]]).unique()
    all_species.sort()  # Sort for deterministic ordering

    le = LabelEncoder()
    le.fit(all_species)

    y_train = le.transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # Extract Feature Matrices with strict float64 precision
    X_train = df_train[feature_names].values.astype(np.float64)
    X_val = df_val[feature_names].values.astype(np.float64)
    X_test = df_test[feature_names].values.astype(np.float64)

    test_ids = df_test["id"].values

    # Get Feature Groups
    feature_groups = get_feature_groups(feature_names)

    print(f"Data Loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {len(classes)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "test_ids": test_ids,
        "classes": classes,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
    }
