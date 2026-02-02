import os
import pandas as pd
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    TARGET_COL,
    DEBUG_SIZE,
    SEED,
)


def load_datasets(load_from_cache=True, debug=False):
    """
    Loads the train, validation, and test datasets.

    Implements caching to parquet files in the working directory.
    Handles debug subsampling and separation of features (X) and target (y).

    Args:
        load_from_cache (bool): If True, attempts to load processed files from the working directory.
        debug (bool): If True, subsamples the datasets to DEBUG_SIZE for faster iteration.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test)
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.
            X_test (pd.DataFrame): Test features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug state
    prefix = "debug_" if debug else ""

    path_X_train = os.path.join(WORKING_DIR, f"{prefix}X_train.parquet")
    path_y_train = os.path.join(WORKING_DIR, f"{prefix}y_train.parquet")
    path_X_val = os.path.join(WORKING_DIR, f"{prefix}X_val.parquet")
    path_y_val = os.path.join(WORKING_DIR, f"{prefix}y_val.parquet")
    path_X_test = os.path.join(WORKING_DIR, f"{prefix}X_test.parquet")

    cache_files = [path_X_train, path_y_train, path_X_val, path_y_val, path_X_test]

    # 1. Try to load from cache
    if load_from_cache and all(os.path.exists(p) for p in cache_files):
        print(f"Loading cached datasets from {WORKING_DIR} (debug={debug})...")
        X_train = pd.read_parquet(path_X_train)
        # Read y as DataFrame then convert to Series for consistency
        y_train = pd.read_parquet(path_y_train)[TARGET_COL]

        X_val = pd.read_parquet(path_X_val)
        y_val = pd.read_parquet(path_y_val)[TARGET_COL]

        X_test = pd.read_parquet(path_X_test)

        return X_train, y_train, X_val, y_val, X_test

    # 2. Load from source metadata
    print(f"Loading raw datasets from metadata (debug={debug})...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    # 3. Apply Debug Subsampling
    if debug:
        print(f"Debug mode active: Subsampling datasets to {DEBUG_SIZE} samples.")
        df_train = df_train.sample(
            n=min(len(df_train), DEBUG_SIZE), random_state=SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), DEBUG_SIZE), random_state=SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), DEBUG_SIZE), random_state=SEED
        ).reset_index(drop=True)

    # 4. Separate Target (y) from Features (X)
    # Train
    if TARGET_COL in df_train.columns:
        y_train = df_train[TARGET_COL]
        X_train = df_train.drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

    # Validation
    if TARGET_COL in df_val.columns:
        y_val = df_val[TARGET_COL]
        X_val = df_val.drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"Target column '{TARGET_COL}' not found in validation data.")

    # Test (No target expected)
    X_test = df_test

    # 5. Save to Cache
    print(f"Caching processed datasets to {WORKING_DIR}...")
    X_train.to_parquet(path_X_train)
    y_train.to_frame().to_parquet(path_y_train)

    X_val.to_parquet(path_X_val)
    y_val.to_frame().to_parquet(path_y_val)

    X_test.to_parquet(path_X_test)

    return X_train, y_train, X_val, y_val, X_test
