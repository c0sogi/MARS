import numpy as np
import pandas as pd
import os
from library.config import Config
from library.features import process_data


def prepare_feature_matrix(metadata_path, cache_path, load_cached_data=True):
    """
    Orchestrates the loading of .xyz files, generation of physical and GNN-based features,
    and merging with tabular data.

    This function wraps the process_data function from the features library which implements
    the specific logic for reading ASE atoms, running the M3GNet model, and caching the results.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_path (str): Path where the processed Parquet file should be stored/loaded.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed feature matrix including targets (if available) and IDs.
    """
    # Delegate to the provided library function which handles ASE loading, GNN inference, and Caching.
    return process_data(metadata_path, cache_path, load_cached_data)


def load_data(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Performs the following steps:
    1. Calls prepare_feature_matrix for each split (Train, Val, Test).
    2. Separates the target variables from the input features.
    3. Drops the 'id' column from the feature set.
    4. Applies log(1+x) transformation to the targets if configured.

    Args:
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        tuple:
            (X_train, y_train): Training features and targets.
            (X_val, y_val): Validation features and targets.
            X_test: Test features.
            test_ids: IDs corresponding to the test set rows.
    """
    print("Loading and preparing datasets...")

    # 1. Generate/Load Feature Matrices
    # These functions handle the heavy lifting of GNN inference and caching
    df_train = prepare_feature_matrix(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_CACHE, load_cached_data
    )
    df_val = prepare_feature_matrix(
        Config.VAL_METADATA_PATH, Config.VAL_FEATURES_CACHE, load_cached_data
    )
    df_test = prepare_feature_matrix(
        Config.TEST_METADATA_PATH, Config.TEST_FEATURES_CACHE, load_cached_data
    )

    # 2. Separate Features (X) and Targets (y)
    target_cols = Config.TARGET_COLS

    # Training Data
    # Ensure targets exist
    for col in target_cols:
        if col not in df_train.columns:
            raise ValueError(f"Target column {col} missing from training data.")

    y_train = df_train[target_cols].copy()
    # Drop targets and ID from features
    X_train = df_train.drop(columns=target_cols + ["id"], errors="ignore")

    # Validation Data
    y_val = df_val[target_cols].copy()
    X_val = df_val.drop(columns=target_cols + ["id"], errors="ignore")

    # Test Data
    # Test set usually does not have targets, but we extract IDs for submission
    if "id" in df_test.columns:
        test_ids = df_test["id"].copy()
        X_test = df_test.drop(columns=["id"], errors="ignore")
    else:
        # Fallback if ID is somehow missing, though metadata generation ensures it
        test_ids = pd.Series(range(len(df_test)), name="id")
        X_test = df_test.copy()

    # 3. Target Transformation
    # Apply log1p transformation to targets to optimize for RMSLE
    if Config.LOG_TRANSFORM_TARGETS:
        print("Applying log1p transformation to target variables...")
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)

    print(f"Data Loaded:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}")

    return (X_train, y_train), (X_val, y_val), X_test, test_ids
