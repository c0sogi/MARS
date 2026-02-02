import os
import pandas as pd
import ase.io
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TARGET_COLS,
)
from library.descriptors import generate_features


def load_geometry(file_path):
    """
    Loads a geometry file using ASE.

    Args:
        file_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The atoms object, or None if reading fails.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    try:
        return ase.io.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None


def process_dataset(load_cached_data=True, debug_limit=None):
    """
    Orchestrates data loading, feature generation, and merging.

    Args:
        load_cached_data (bool): If True, attempts to load features from parquet cache.
        debug_limit (int, optional): If set, limits the number of samples for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(VAL_METADATA_PATH)
    test_meta = pd.read_csv(TEST_METADATA_PATH)

    # 2. Generate Features (Caching logic is handled within generate_features)
    print("Processing Training Data...")
    train_feats = generate_features(
        train_meta,
        load_cached_data=load_cached_data,
        cache_path=TRAIN_FEATURES_PATH,
        debug_limit=debug_limit,
    )

    print("Processing Validation Data...")
    val_feats = generate_features(
        val_meta,
        load_cached_data=load_cached_data,
        cache_path=VAL_FEATURES_PATH,
        debug_limit=debug_limit,
    )

    print("Processing Test Data...")
    test_feats = generate_features(
        test_meta,
        load_cached_data=load_cached_data,
        cache_path=TEST_FEATURES_PATH,
        debug_limit=debug_limit,
    )

    # 3. Merge Features with Metadata (Tabular data)
    # Use inner join to ensure we only keep rows where features were successfully generated
    train_full = pd.merge(train_meta, train_feats, on="id", how="inner")
    val_full = pd.merge(val_meta, val_feats, on="id", how="inner")
    test_full = pd.merge(test_meta, test_feats, on="id", how="inner")

    # 4. Clean Data
    # Remove file_path if present as it's not a feature
    for df in [train_full, val_full, test_full]:
        if "file_path" in df.columns:
            df.drop(columns=["file_path"], inplace=True)

    # 5. Prepare Output
    # Targets
    y_train = train_full[TARGET_COLS]
    y_val = val_full[TARGET_COLS]

    # Features: Drop ID and Targets from training/validation sets
    drop_cols_train = ["id"] + TARGET_COLS
    X_train = train_full.drop(columns=drop_cols_train, errors="ignore")

    drop_cols_val = ["id"] + TARGET_COLS
    X_val = val_full.drop(columns=drop_cols_val, errors="ignore")

    # Test Features: Drop ID (test set doesn't have targets)
    test_ids = test_full["id"]
    X_test = test_full.drop(columns=["id"], errors="ignore")

    # 6. Align Columns
    # Ensure validation and test sets have the same columns as the training set.
    # This handles cases where certain element pairs might be missing in smaller splits.
    train_cols = X_train.columns.tolist()

    X_val = X_val.reindex(columns=train_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=train_cols, fill_value=0.0)

    print(f"Data Processed. Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
