import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    FLOAT_PRECISION,
    TABULAR_FEATURE_PREFIXES,
    SEED,
)
from library.image_features import process_dataset

# Ensure reproducibility
np.random.seed(SEED)


def _get_tabular_features(df):
    """
    Extracts tabular feature columns from the metadata DataFrame.

    Args:
        df (pd.DataFrame): Metadata DataFrame containing all columns.

    Returns:
        pd.DataFrame: DataFrame with 'id' and tabular feature columns.
    """
    # Identify columns that start with the defined prefixes (margin, shape, texture)
    feature_cols = []
    for col in df.columns:
        for prefix in TABULAR_FEATURE_PREFIXES:
            if col.startswith(prefix):
                feature_cols.append(col)
                break

    # Return id and features
    return df[["id"] + feature_cols].copy()


def load_and_merge_data(dataset_name, metadata_path, load_cached_data=True):
    """
    Loads metadata, extracts/loads image features, merges with tabular features,
    and returns X, y, and ids. Implements caching of the merged result.

    Args:
        dataset_name (str): 'train', 'val', or 'test'.
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to use cached merged data if available.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Feature matrix (float64).
            y (np.ndarray or None): Target labels (strings) or None if test set.
            ids (np.ndarray): Image IDs.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_merged.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading merged data for '{dataset_name}' from {cache_file}")
        try:
            df_merged = pd.read_parquet(cache_file)

            # Extract components
            ids = df_merged["id"].values

            if "species" in df_merged.columns:
                y = df_merged["species"].values
                X = df_merged.drop(columns=["id", "species"])
            else:
                y = None
                X = df_merged.drop(columns=["id"])

            # Ensure float64 precision for features
            X = X.astype(FLOAT_PRECISION)

            return X, y, ids
        except Exception as e:
            print(f"Failed to load merged cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data for '{dataset_name}'...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Get Image Features (Geometry + Internal Structure)
    # This function handles its own caching of the image extraction process
    df_image_feats = process_dataset(
        df_meta, dataset_name, load_cached_data=load_cached_data
    )

    # Get Tabular Features (Margin, Shape, Texture)
    df_tabular_feats = _get_tabular_features(df_meta)

    # Merge on ID
    # df_meta contains 'species' (target), so we start with that (if it exists)
    cols_to_keep = ["id"]
    if "species" in df_meta.columns:
        cols_to_keep.append("species")

    df_base = df_meta[cols_to_keep].copy()

    # Merge tabular features
    df_merged = pd.merge(df_base, df_tabular_feats, on="id", how="left")

    # Merge image features
    df_merged = pd.merge(df_merged, df_image_feats, on="id", how="left")

    # 3. Post-processing

    # Separate columns
    id_col = "id"
    target_col = "species" if "species" in df_merged.columns else None

    # Identify feature columns (all columns except id and species)
    exclude = {id_col}
    if target_col:
        exclude.add(target_col)

    feature_cols = [c for c in df_merged.columns if c not in exclude]

    # Enforce Alphanumeric Sorting of feature columns
    feature_cols.sort()

    # Reorder DataFrame
    final_cols = [id_col]
    if target_col:
        final_cols.append(target_col)
    final_cols.extend(feature_cols)

    df_merged = df_merged[final_cols]

    # Fill NaNs with 0 (though there shouldn't be any if pipelines are correct)
    # Image extraction returns 0.0 on failure, tabular should be complete.
    df_merged[feature_cols] = df_merged[feature_cols].fillna(0.0)

    # 4. Save to cache
    print(f"Saving merged data for '{dataset_name}' to {cache_file}")
    df_merged.to_parquet(cache_file, index=False)

    # 5. Return components
    ids = df_merged[id_col].values

    if target_col:
        y = df_merged[target_col].values
        X = df_merged[feature_cols]
    else:
        y = None
        X = df_merged[feature_cols]

    # Ensure precision
    X = X.astype(FLOAT_PRECISION)

    return X, y, ids


def get_train_data(load_cached_data=True):
    """Wrapper to get training data."""
    return load_and_merge_data("train", TRAIN_METADATA_PATH, load_cached_data)


def get_val_data(load_cached_data=True):
    """Wrapper to get validation data."""
    return load_and_merge_data("val", VAL_METADATA_PATH, load_cached_data)


def get_test_data(load_cached_data=True):
    """Wrapper to get test data."""
    return load_and_merge_data("test", TEST_METADATA_PATH, load_cached_data)
