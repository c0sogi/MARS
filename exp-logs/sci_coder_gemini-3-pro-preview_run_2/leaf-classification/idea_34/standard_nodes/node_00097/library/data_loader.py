import pandas as pd
import numpy as np
import os
from library import config
from library import features


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, separating provided 'Global' features from extracted 'Macro' features.
    Enforces float64 precision and handles caching for expensive feature extraction.

    Args:
        load_cached_data (bool): If True, attempts to load macro features from cache.

    Returns:
        dict: A dictionary containing:
            'train': {'X_global', 'X_macro', 'y'}
            'val':   {'X_global', 'X_macro', 'y'}
            'test':  {'X_global', 'X_macro', 'ids'}
            'classes': List of unique species names sorted alphabetically.
    """
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # -------------------------------------------------------------------------
    # 1. Process Global Features (Provided 192 features)
    # -------------------------------------------------------------------------
    print("Processing Global features (Margin, Shape, Texture)...")

    # Identify feature columns by excluding metadata columns
    # Train/Val have 'species', Test does not.
    non_feature_cols_train = ["id", "species", "image_path"]
    non_feature_cols_test = ["id", "image_path"]

    # Extract Global Features for Train
    global_cols_train = [c for c in df_train.columns if c not in non_feature_cols_train]
    X_train_global = df_train[global_cols_train].values.astype(config.FLOAT_PRECISION)

    # Extract Global Features for Val
    global_cols_val = [c for c in df_val.columns if c not in non_feature_cols_train]
    X_val_global = df_val[global_cols_val].values.astype(config.FLOAT_PRECISION)

    # Extract Global Features for Test
    global_cols_test = [c for c in df_test.columns if c not in non_feature_cols_test]
    X_test_global = df_test[global_cols_test].values.astype(config.FLOAT_PRECISION)

    # Verify feature counts
    assert (
        X_train_global.shape[1] == 192
    ), f"Expected 192 global features, got {X_train_global.shape[1]}"
    assert (
        X_test_global.shape[1] == 192
    ), f"Expected 192 global features, got {X_test_global.shape[1]}"

    # -------------------------------------------------------------------------
    # 2. Process Macro Features (Extracted from Images)
    # -------------------------------------------------------------------------
    print("Processing Macro features (Morphometrics)...")

    # Train Macro Features
    df_train_macro = features.get_macro_features(
        df_train, config.CACHE_TRAIN_MACRO, load_cached_data=load_cached_data
    )
    X_train_macro = df_train_macro.values.astype(config.FLOAT_PRECISION)

    # Val Macro Features
    df_val_macro = features.get_macro_features(
        df_val, config.CACHE_VAL_MACRO, load_cached_data=load_cached_data
    )
    X_val_macro = df_val_macro.values.astype(config.FLOAT_PRECISION)

    # Test Macro Features
    df_test_macro = features.get_macro_features(
        df_test, config.CACHE_TEST_MACRO, load_cached_data=load_cached_data
    )
    X_test_macro = df_test_macro.values.astype(config.FLOAT_PRECISION)

    # -------------------------------------------------------------------------
    # 3. Process Targets and IDs
    # -------------------------------------------------------------------------
    print("Processing targets and IDs...")

    y_train = df_train["species"].values
    y_val = df_val["species"].values
    test_ids = df_test["id"].values

    # Derive unique classes from training data (sorted alphabetically)
    classes = np.sort(np.unique(y_train))

    print(f"Data loaded successfully.")
    print(f"Train shapes: Global {X_train_global.shape}, Macro {X_train_macro.shape}")
    print(f"Val shapes:   Global {X_val_global.shape}, Macro {X_val_macro.shape}")
    print(f"Test shapes:  Global {X_test_global.shape}, Macro {X_test_macro.shape}")
    print(f"Number of classes: {len(classes)}")

    return {
        "train": {"X_global": X_train_global, "X_macro": X_train_macro, "y": y_train},
        "val": {"X_global": X_val_global, "X_macro": X_val_macro, "y": y_val},
        "test": {"X_global": X_test_global, "X_macro": X_test_macro, "ids": test_ids},
        "classes": classes,
    }
