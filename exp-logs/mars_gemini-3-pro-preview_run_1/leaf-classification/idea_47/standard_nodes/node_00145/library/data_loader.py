import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    WORKING_DIR,
    FLOAT_PRECISION,
    TABULAR_FEATURE_PREFIXES,
)
from library.image_features import batch_extract


def _process_split(metadata_path, split_name, load_cached_data=True):
    """
    Internal helper to process a single data split (train, val, or test).
    Handles loading, feature extraction, fusion, sorting, and caching.
    """
    # Define cache path for the fused dataset
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_fused.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_fused = pd.read_parquet(cache_path)
            # Separate components
            ids = df_fused["id"].values

            # Identify feature columns (exclude metadata columns)
            exclude_cols = ["id", "species"]
            feature_cols = [c for c in df_fused.columns if c not in exclude_cols]

            # Ensure alphanumeric order is preserved (parquet usually preserves, but we enforce)
            feature_cols = sorted(feature_cols)

            X = df_fused[feature_cols].values.astype(FLOAT_PRECISION)

            if "species" in df_fused.columns:
                y = df_fused["species"].values
                return X, y, ids
            else:
                return X, ids
        except Exception as e:
            print(f"Cache load failed for {split_name}: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # A. Extract Tabular Features
    # Filter columns that start with the defined prefixes
    tabular_cols = [
        col
        for col in df_meta.columns
        if any(col.startswith(prefix) for prefix in TABULAR_FEATURE_PREFIXES)
    ]
    df_tabular = df_meta[tabular_cols].copy()

    # B. Extract Geometric Features
    # Construct full image paths
    # metadata 'file_path' is relative (e.g., "images/1.jpg")
    full_image_paths = (
        df_meta["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x)).tolist()
    )

    # Batch extract (handles its own caching for the image processing part)
    df_geo = batch_extract(full_image_paths, load_cached_data=load_cached_data)

    # C. Fusion
    # Concatenate tabular and geometric features
    # Reset indices to ensure alignment before concat
    df_tabular.reset_index(drop=True, inplace=True)
    df_geo.reset_index(drop=True, inplace=True)

    df_features = pd.concat([df_tabular, df_geo], axis=1)

    # D. Deterministic Ordering
    # Sort columns alphanumerically
    sorted_cols = sorted(df_features.columns)
    df_features = df_features[sorted_cols]

    # E. Prepare for Cache Saving
    # Create a dataframe that includes ID and Species (if available) for storage
    df_save = df_features.copy()
    df_save["id"] = df_meta["id"].values

    if "species" in df_meta.columns:
        df_save["species"] = df_meta["species"].values

    # Save to parquet
    try:
        df_save.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save fused cache to {cache_path}: {e}")

    # F. Return formatted arrays
    X = df_features.values.astype(FLOAT_PRECISION)
    ids = df_meta["id"].values

    if "species" in df_meta.columns:
        y = df_meta["species"].values
        return X, y, ids
    else:
        return X, ids


def load_data(load_cached_data=True):
    """
    Main entry point to load all datasets.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: ((X_train, y_train, ids_train),
                (X_val, y_val, ids_val),
                (X_test, ids_test))
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Process Train
    X_train, y_train, ids_train = _process_split(
        TRAIN_METADATA_PATH, "train", load_cached_data=load_cached_data
    )

    # Process Val
    X_val, y_val, ids_val = _process_split(
        VAL_METADATA_PATH, "val", load_cached_data=load_cached_data
    )

    # Process Test
    X_test, ids_test = _process_split(
        TEST_METADATA_PATH, "test", load_cached_data=load_cached_data
    )

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)
