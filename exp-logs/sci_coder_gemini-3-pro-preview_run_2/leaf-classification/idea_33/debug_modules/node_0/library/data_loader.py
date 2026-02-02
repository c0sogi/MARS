import os
import numpy as np
import pandas as pd
from library import config
from library import feature_extraction


def get_feature_groups():
    """
    Returns a dictionary mapping view names to list of column names.
    Used by the model to select specific feature subsets for different experts.
    """
    # Global View: The 192 pre-extracted features
    global_cols = []
    for prefix in config.FEATURE_PREFIXES:
        for i in range(1, config.NUM_FEATURES_PER_GROUP + 1):
            global_cols.append(f"{prefix}_{i}")

    # Macro View: The 11 extracted macro features
    macro_cols = config.MACRO_FEATURE_NAMES

    return {"global": global_cols, "macro": macro_cols, "all": global_cols + macro_cols}


def load_and_merge_features(dataset_key, load_cached_data=True):
    """
    Loads metadata, extracts/loads macro features, merges them, and handles caching of the merged dataframe.

    Args:
        dataset_key (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        pd.DataFrame: The merged dataframe containing all features and metadata.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.CACHE_DIR, f"{dataset_key}_merged.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure precision is correct after loading (Parquet might default to float32 sometimes)
            feature_info = get_feature_groups()
            all_feats = feature_info["all"]
            # Only cast columns that exist in the dataframe
            cols_to_cast = [c for c in all_feats if c in df.columns]
            df[cols_to_cast] = df[cols_to_cast].astype(config.FLOAT_PRECISION)
            return df
        except Exception as e:
            print(f"Failed to load merged cache for {dataset_key}: {e}. Recomputing...")

    # 2. Load Metadata
    if dataset_key == "train":
        metadata_path = config.TRAIN_METADATA_PATH
    elif dataset_key == "val":
        metadata_path = config.VAL_METADATA_PATH
    elif dataset_key == "test":
        metadata_path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid dataset_key: {dataset_key}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # 3. Extract/Load Macro Features
    # Note: feature_extraction.extract_macro_features handles its own caching of the extraction process
    df_macro = feature_extraction.extract_macro_features(
        df_meta, dataset_key, load_cached_data=load_cached_data
    )

    # 4. Merge
    # df_meta has 'id', 'species' (if train/val), 'image_path', and the 192 pre-extracted features
    # df_macro has 'id' and the 11 macro features
    # We merge on 'id'
    df_merged = pd.merge(df_meta, df_macro, on="id", how="left")

    # 5. Enforce Precision on Feature Columns
    feature_info = get_feature_groups()
    all_feature_cols = feature_info["all"]

    # Validate that all expected columns are present
    missing_cols = [c for c in all_feature_cols if c not in df_merged.columns]
    if missing_cols:
        print(
            f"Warning: The following feature columns are missing after merge: {missing_cols}"
        )

    # Cast to float64 (double precision)
    for col in all_feature_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(config.FLOAT_PRECISION)

    # 6. Save Cache
    try:
        df_merged.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save merged cache to {cache_path}: {e}")

    return df_merged


def get_data_splits(load_cached_data=True):
    """
    Returns the training and validation splits (Phase 1: Selection).

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
            X_train, X_val: pd.DataFrame with all features.
            y_train, y_val: np.array of species strings.
    """
    # Load Train
    df_train = load_and_merge_features("train", load_cached_data)
    # Load Val
    df_val = load_and_merge_features("val", load_cached_data)

    # Define Feature Columns (exclude metadata)
    feature_cols = get_feature_groups()["all"]
    # Filter to only those present
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["species"].values

    X_val = df_val[feature_cols].copy()
    y_val = df_val["species"].values

    return X_train, y_train, X_val, y_val


def get_full_train_data(load_cached_data=True):
    """
    Returns the combined training and validation data (Phase 2: Final Retraining).

    Returns:
        tuple: (X_full, y_full)
    """
    # Load both
    df_train = load_and_merge_features("train", load_cached_data)
    df_val = load_and_merge_features("val", load_cached_data)

    # Concatenate
    df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    feature_cols = get_feature_groups()["all"]
    feature_cols = [c for c in feature_cols if c in df_full.columns]

    X_full = df_full[feature_cols].copy()
    y_full = df_full["species"].values

    return X_full, y_full


def get_test_data(load_cached_data=True):
    """
    Returns the test data for inference.

    Returns:
        tuple: (X_test, test_ids)
            X_test: pd.DataFrame with all features.
            test_ids: np.array of image IDs.
    """
    df_test = load_and_merge_features("test", load_cached_data)

    feature_cols = get_feature_groups()["all"]
    feature_cols = [c for c in feature_cols if c in df_test.columns]

    X_test = df_test[feature_cols].copy()
    test_ids = df_test["id"].values

    return X_test, test_ids
