import os
import pandas as pd
import numpy as np
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    FLOAT_PRECISION,
    GEOMETRIC_FEATURES,
    TABULAR_FEATURE_GROUPS,
    ID_COL,
    TARGET_COL,
)
from library.utils import get_config_hash
from library.geometry import get_geometric_features


def get_merge_config_hash():
    """
    Generates a hash based on the feature configuration to ensure
    cache validity when feature definitions change.
    """
    config = {
        "geometric_features": GEOMETRIC_FEATURES,
        "tabular_groups": TABULAR_FEATURE_GROUPS,
        "sort_order": "alphanumeric",
        "precision": str(FLOAT_PRECISION),
    }
    return get_config_hash(config)


def load_and_merge_data(dataset_name, load_cached_data=True):
    """
    Loads metadata, extracts/loads geometric features, merges them with tabular features,
    and returns the prepared data matrices.

    Args:
        dataset_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (X, y, ids)
            - X (pd.DataFrame): Feature matrix (float64), alphanumerically sorted columns.
            - y (np.ndarray or None): Target labels (strings) if available, else None.
            - ids (np.ndarray): Image IDs.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache filename with config hash
    config_hash = get_merge_config_hash()
    cache_filename = f"{dataset_name}_merged_{config_hash}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading merged {dataset_name} data from {cache_path}...")
        try:
            df_merged = pd.read_parquet(cache_path)

            # Extract components
            ids = df_merged[ID_COL].values

            if TARGET_COL in df_merged.columns:
                y = df_merged[TARGET_COL].values
                drop_cols = [ID_COL, TARGET_COL]
            else:
                y = None
                drop_cols = [ID_COL]

            # X is everything else
            X = df_merged.drop(columns=drop_cols)

            # Enforce precision just in case
            X = X.astype(FLOAT_PRECISION)

            return X, y, ids
        except Exception as e:
            print(f"Failed to load merged cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing and merging data for {dataset_name}...")

    # Load Metadata
    metadata_path = os.path.join(METADATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Load Geometric Features (delegates caching to geometry module)
    df_geo = get_geometric_features(
        metadata_path, dataset_name, load_cached_data=load_cached_data
    )

    # Merge on ID
    # df_meta contains: id, species (optional), file_path, margin_*, shape_*, texture_*
    # df_geo contains: id, Area, Mean_Thickness, ...
    df_full = pd.merge(df_meta, df_geo, on=ID_COL, how="left")

    # Identify Feature Columns
    # 1. Geometric Features
    geo_cols = GEOMETRIC_FEATURES

    # 2. Tabular Features (filter from metadata columns)
    # We look for columns starting with the group names defined in config
    tab_cols = [
        c
        for c in df_meta.columns
        if any(c.startswith(prefix) for prefix in TABULAR_FEATURE_GROUPS)
    ]

    all_feature_cols = geo_cols + tab_cols

    # Enforce Alphanumeric Column Ordering
    # This is crucial for deterministic behavior in the linear model
    all_feature_cols = sorted(all_feature_cols)

    # Extract X
    X = df_full[all_feature_cols].copy()

    # Enforce Float64 Precision
    X = X.astype(FLOAT_PRECISION)

    # Extract IDs
    ids = df_full[ID_COL].values

    # Extract Target if present
    if TARGET_COL in df_full.columns:
        y = df_full[TARGET_COL].values
    else:
        y = None

    # 3. Save to Cache
    # We construct a dataframe that includes ID and Target (if exists) + Features
    # to save as a single parquet file.
    df_to_save = X.copy()
    df_to_save.insert(0, ID_COL, ids)
    if y is not None:
        df_to_save.insert(1, TARGET_COL, y)

    try:
        df_to_save.to_parquet(cache_path, index=False)
        print(f"Saved merged data to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save merged cache: {e}")

    return X, y, ids
