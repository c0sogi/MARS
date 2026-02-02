import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_cache, load_cache, load_metadata
from library.image_features import process_images


def get_class_names(load_cached_data=True):
    """
    Returns the sorted list of unique species names from the training set.
    Ensures consistent class ordering for encoding and submission.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        np.ndarray: Sorted array of class names.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_names.npy")

    if load_cached_data:
        cached = load_cache(cache_path)
        if cached is not None:
            return cached

    # Load train metadata to get classes
    print("Extracting class names from training metadata...")
    df = load_metadata(Config.TRAIN_METADATA_PATH)
    classes = np.sort(df["species"].unique())

    save_cache(classes, cache_path)
    return classes


def load_dataset(dataset_name, load_cached_data=True):
    """
    Loads the specified dataset (train, val, or test).
    Handles caching of the structured feature matrices and labels.

    Args:
        dataset_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        dict: Dictionary containing:
            - 'X_global': np.ndarray (N, 192) - Provided features
            - 'X_morph': np.ndarray (N, 11) - Extracted image features
            - 'y': np.ndarray (N,) - Species labels (strings), None for test
            - 'ids': np.ndarray (N,) - Image IDs
    """
    valid_names = ["train", "val", "test"]
    if dataset_name not in valid_names:
        raise ValueError(f"dataset_name must be one of {valid_names}")

    # Define cache paths for the structured components
    cache_X_global = os.path.join(Config.CACHE_DIR, f"{dataset_name}_X_global.npy")
    cache_X_morph = os.path.join(Config.CACHE_DIR, f"{dataset_name}_X_morph.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"{dataset_name}_ids.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"{dataset_name}_y.npy")

    # Determine if we expect targets (y)
    expect_y = dataset_name in ["train", "val"]

    # 1. Try Loading Cache
    if load_cached_data:
        X_global = load_cache(cache_X_global)
        X_morph = load_cache(cache_X_morph)
        ids = load_cache(cache_ids)
        y = load_cache(cache_y) if expect_y else None

        # Check if all required parts were loaded successfully
        all_loaded = (
            (X_global is not None) and (X_morph is not None) and (ids is not None)
        )
        if expect_y:
            all_loaded = all_loaded and (y is not None)

        if all_loaded:
            print(f"Loaded {dataset_name} dataset from cache.")
            return {"X_global": X_global, "X_morph": X_morph, "y": y, "ids": ids}

    # 2. Process from Scratch
    print(f"Processing {dataset_name} dataset from source metadata...")

    # Identify metadata file
    if dataset_name == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif dataset_name == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df = load_metadata(meta_path)

    # Extract IDs
    ids = df["id"].values.astype(int)

    # Extract Targets (if applicable)
    y = None
    if expect_y:
        y = df["species"].values

    # Extract Global Features (192 columns)
    # We strictly select columns based on the known prefixes to avoid id/species/path
    # The columns are named margin_1..64, shape_1..64, texture_1..64
    feat_cols = []
    prefixes = ["margin", "shape", "texture"]

    # We iterate to preserve order: margin -> shape -> texture
    for prefix in prefixes:
        for i in range(1, 65):
            col_name = f"{prefix}_{i}"
            if col_name in df.columns:
                feat_cols.append(col_name)
            else:
                # Fallback: if specific indices are missing, try strict filtering
                # This handles cases where columns might be named differently but usually matches
                pass

    # If explicit construction failed (unlikely), use list comprehension
    if len(feat_cols) != Config.N_PROVIDED_FEATURES:
        feat_cols = [c for c in df.columns if c.startswith(tuple(prefixes))]

    X_global = df[feat_cols].values.astype(Config.FLOAT_TYPE)

    # Extract Morphometric Features
    # Delegates to image_features module which handles reading images and its own caching
    X_morph = process_images(df, dataset_name, load_cached_data=load_cached_data)

    # 3. Save Cache
    save_cache(X_global, cache_X_global)
    save_cache(X_morph, cache_X_morph)
    save_cache(ids, cache_ids)
    if expect_y:
        save_cache(y, cache_y)

    return {"X_global": X_global, "X_morph": X_morph, "y": y, "ids": ids}
