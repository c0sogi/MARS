import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    TARGET_COL,
    ID_COL,
)
from library.features import extract_morphometrics


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Merges metadata features with extracted morphometric features.
    Implements caching for the merged datasets using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "train_merged.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "val_merged.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "test_merged.parquet")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    ):
        print("Loading merged datasets from cache...")
        df_train = pd.read_parquet(cache_train_path)
        df_val = pd.read_parquet(cache_val_path)
        df_test = pd.read_parquet(cache_test_path)
        return df_train, df_val, df_test

    print("Processing datasets from scratch...")

    # 2. Load Metadata
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {TRAIN_DATA_PATH}")

    df_train_meta = pd.read_csv(TRAIN_DATA_PATH)
    df_val_meta = pd.read_csv(VAL_DATA_PATH)
    df_test_meta = pd.read_csv(TEST_DATA_PATH)

    # 3. Extract Morphometrics
    # The extract_morphometrics function handles its own caching for the extraction step.
    # We pass the load_cached_data flag down.
    df_train_morph = extract_morphometrics(
        df_train_meta, "train", load_cached_data=load_cached_data
    )
    df_val_morph = extract_morphometrics(
        df_val_meta, "val", load_cached_data=load_cached_data
    )
    df_test_morph = extract_morphometrics(
        df_test_meta, "test", load_cached_data=load_cached_data
    )

    # 4. Merge Datasets
    # Merge on ID to combine provided features with extracted morphometrics
    df_train = pd.merge(df_train_meta, df_train_morph, on=ID_COL, how="inner")
    df_val = pd.merge(df_val_meta, df_val_morph, on=ID_COL, how="inner")
    df_test = pd.merge(df_test_meta, df_test_morph, on=ID_COL, how="inner")

    # 5. Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    df_train.to_parquet(cache_train_path, index=False)
    df_val.to_parquet(cache_val_path, index=False)
    df_test.to_parquet(cache_test_path, index=False)

    print(f"Datasets processed and merged. Cached at {WORKING_DIR}")

    return df_train, df_val, df_test


def get_feature_columns(df):
    """
    Helper to identify feature columns by excluding metadata columns.
    """
    exclude_cols = [ID_COL, TARGET_COL, "image_path"]
    return [c for c in df.columns if c not in exclude_cols]


def get_train_val_data(load_cached_data=True):
    """
    Returns (X_train, y_train, X_val, y_val) for the selection phase.
    """
    df_train, df_val, _ = load_datasets(load_cached_data=load_cached_data)

    feature_cols = get_feature_columns(df_train)

    X_train = df_train[feature_cols]
    y_train = df_train[TARGET_COL]

    X_val = df_val[feature_cols]
    y_val = df_val[TARGET_COL]

    return X_train, y_train, X_val, y_val


def get_full_train_data(load_cached_data=True):
    """
    Returns (X_full, y_full) combining train and val for the final retraining phase.
    """
    df_train, df_val, _ = load_datasets(load_cached_data=load_cached_data)

    # Concatenate train and val
    df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    feature_cols = get_feature_columns(df_full)

    X_full = df_full[feature_cols]
    y_full = df_full[TARGET_COL]

    return X_full, y_full


def get_test_data(load_cached_data=True):
    """
    Returns (X_test, test_ids) for inference.
    """
    _, _, df_test = load_datasets(load_cached_data=load_cached_data)

    feature_cols = get_feature_columns(df_test)

    X_test = df_test[feature_cols]
    ids_test = df_test[ID_COL]

    return X_test, ids_test
