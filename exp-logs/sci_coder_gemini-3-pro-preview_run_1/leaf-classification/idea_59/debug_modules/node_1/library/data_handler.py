import os
import pandas as pd
import numpy as np
from library import config, utils, image_features


def _get_tabular_columns():
    """
    Generates the list of pre-extracted tabular feature column names
    (margin_1...64, shape_1...64, texture_1...64).
    """
    cols = []
    for prefix in config.TABULAR_FEATURE_PREFIXES:
        for i in range(1, config.NUM_TABULAR_FEATURES_PER_SET + 1):
            cols.append(f"{prefix}{i}")
    return cols


def _get_cache_paths(dataset_type, debug_suffix):
    """
    Generates file paths for caching X, y, and ids.
    """
    base_name = f"{dataset_type}{debug_suffix}"
    x_path = os.path.join(config.CACHE_DIR, f"X_{base_name}.parquet")
    y_path = os.path.join(config.CACHE_DIR, f"y_{base_name}.npy")
    ids_path = os.path.join(config.CACHE_DIR, f"ids_{base_name}.npy")
    return x_path, y_path, ids_path


def load_dataset(dataset_type, debug_size=None, load_cached_data=True):
    """
    Loads the dataset, merging tabular features with computed geometric features.
    Handles caching, debugging (downsampling), and deterministic sorting.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        debug_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Feature matrix (float64).
            y (np.ndarray): Target labels (strings) or None for test set.
            ids (np.ndarray): Image IDs.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Determine debug suffix for cache isolation
    debug_suffix = f"_debug_{debug_size}" if debug_size is not None else ""

    # Get cache paths
    x_path, y_path, ids_path = _get_cache_paths(dataset_type, debug_suffix)

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if X and ids exist (y is optional for test)
        if os.path.exists(x_path) and os.path.exists(ids_path):
            # For train/val, y must also exist
            if dataset_type in ["train", "val"] and not os.path.exists(y_path):
                pass  # Cache incomplete, proceed to compute
            else:
                print(
                    f"Loading {dataset_type} dataset from cache ({config.CACHE_DIR})..."
                )
                X = pd.read_parquet(x_path)
                ids = np.load(ids_path)

                y = None
                if os.path.exists(y_path):
                    y = np.load(y_path, allow_pickle=True)

                return X, y, ids

    # 2. Compute from Scratch
    print(f"Processing {dataset_type} dataset from metadata...")

    # Load Metadata
    if dataset_type == "train":
        meta_path = config.TRAIN_CSV
    elif dataset_type == "val":
        meta_path = config.VAL_CSV
    elif dataset_type == "test":
        meta_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Apply Debug Sampling
    if debug_size is not None:
        print(f"Debug Mode: Sampling {debug_size} rows from {dataset_type}.")
        if len(df_meta) > debug_size:
            # Use fixed seed for reproducible sampling
            df_meta = df_meta.sample(
                n=debug_size, random_state=config.SEED
            ).reset_index(drop=True)

    # Extract IDs
    ids = df_meta["id"].values

    # Extract Target (if present)
    y = None
    if "species" in df_meta.columns:
        y = df_meta["species"].values

    # --- Feature Engineering ---

    # A. Tabular Features
    tabular_cols = _get_tabular_columns()
    # Ensure these columns exist
    missing_cols = [c for c in tabular_cols if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing tabular columns in metadata: {missing_cols[:5]}...")

    X_tabular = df_meta[tabular_cols].astype(config.FLOAT_PRECISION)

    # B. Geometric Features (computed from images)
    # Pass the specific dataset name + suffix to image_features to avoid cache collisions
    # between full runs and debug runs.
    img_dataset_name = f"{dataset_type}{debug_suffix}"
    X_geometric = image_features.extract_integral_features(
        df_meta, dataset_name=img_dataset_name, load_cached_data=load_cached_data
    )

    # C. Merge
    # Reset indices to ensure clean concatenation
    X_tabular.reset_index(drop=True, inplace=True)
    X_geometric.reset_index(drop=True, inplace=True)

    X = pd.concat([X_tabular, X_geometric], axis=1)

    # D. Sanitization
    # 1. Enforce float64
    X = X.astype(config.FLOAT_PRECISION)

    # 2. Deterministic Column Sorting (Alphanumeric)
    # This ensures that the order of features is always the same,
    # regardless of how they were concatenated.
    sorted_cols = sorted(X.columns.tolist())
    X = X[sorted_cols]

    # 3. Save to Cache
    print(f"Caching processed {dataset_type} data to {config.CACHE_DIR}...")
    X.to_parquet(x_path, index=False)
    np.save(ids_path, ids)

    if y is not None:
        np.save(y_path, y)

    return X, y, ids
