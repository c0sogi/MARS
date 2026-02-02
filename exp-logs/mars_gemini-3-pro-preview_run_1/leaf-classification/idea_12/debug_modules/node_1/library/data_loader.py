import os
import pandas as pd
import numpy as np
from library.config import Config


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    This function implements a caching mechanism to speed up subsequent runs.
    It strictly enforces the feature order defined in Config.FEATURES to ensure
    deterministic behavior for downstream matrix operations.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the cache directory defined in Config.

    Returns:
        tuple: A tuple containing three tuples:
            (X_train, y_train, ids_train): Training features (DataFrame), targets (Array), IDs (Array)
            (X_val, y_val, ids_val): Validation features (DataFrame), targets (Array), IDs (Array)
            (X_test, ids_test): Test features (DataFrame), IDs (Array)
    """
    # 1. Setup Cache Directory
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Define Cache File Paths
    # We use Parquet for feature matrices to preserve column names and ensure schema consistency.
    # We use NPY for 1D arrays (targets and IDs) for efficiency.
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.parquet"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "ids_train": os.path.join(cache_dir, "ids_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.parquet"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "ids_val": os.path.join(cache_dir, "ids_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.parquet"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # 3. Attempt to Load from Cache
    if load_cached_data:
        # Check if all required files exist
        if all(os.path.exists(p) for p in cache_files.values()):
            print(f"Loading datasets from cache at {cache_dir}...")
            try:
                X_train = pd.read_parquet(cache_files["X_train"])
                # allow_pickle=True is used to support loading string arrays if they were saved as objects,
                # though we attempt to save them as unicode to avoid this dependency.
                y_train = np.load(cache_files["y_train"], allow_pickle=True)
                ids_train = np.load(cache_files["ids_train"], allow_pickle=True)

                X_val = pd.read_parquet(cache_files["X_val"])
                y_val = np.load(cache_files["y_val"], allow_pickle=True)
                ids_val = np.load(cache_files["ids_val"], allow_pickle=True)

                X_test = pd.read_parquet(cache_files["X_test"])
                ids_test = np.load(cache_files["ids_test"], allow_pickle=True)

                return (
                    (X_train, y_train, ids_train),
                    (X_val, y_val, ids_val),
                    (X_test, ids_test),
                )
            except Exception as e:
                print(
                    f"Cache load failed: {e}. Falling back to source data processing."
                )
        else:
            print("Cache incomplete or missing. Processing source data...")
    else:
        print("Skipping cache as requested. Processing source data...")

    # 4. Load Raw Data from Metadata
    print("Loading metadata files...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata missing: {Config.TRAIN_METADATA_PATH}")
    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(f"Val metadata missing: {Config.VAL_METADATA_PATH}")
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata missing: {Config.TEST_METADATA_PATH}")

    df_train_raw = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_raw = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_raw = pd.read_csv(Config.TEST_METADATA_PATH)

    # 5. Enforce Deterministic Feature Schema
    # We select columns explicitly using the sorted list from Config.FEATURES.
    # This guarantees that X_train, X_val, and X_test have the exact same column order.
    features = Config.FEATURES

    # Check for missing columns
    missing_cols = [col for col in features if col not in df_train_raw.columns]
    if missing_cols:
        raise ValueError(
            f"The following required features are missing from training data: {missing_cols}"
        )

    print("Processing and reordering features...")

    # Train Set
    X_train = df_train_raw[features].copy()
    # Convert to string (unicode) explicitly to avoid object-array pickling issues
    y_train = df_train_raw["species"].values.astype(str)
    ids_train = df_train_raw["id"].values

    # Validation Set
    X_val = df_val_raw[features].copy()
    y_val = df_val_raw["species"].values.astype(str)
    ids_val = df_val_raw["id"].values

    # Test Set
    X_test = df_test_raw[features].copy()
    ids_test = df_test_raw["id"].values

    # 6. Save to Cache
    print(f"Saving processed datasets to {cache_dir}...")

    X_train.to_parquet(cache_files["X_train"], index=False)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["ids_train"], ids_train)

    X_val.to_parquet(cache_files["X_val"], index=False)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["ids_val"], ids_val)

    X_test.to_parquet(cache_files["X_test"], index=False)
    np.save(cache_files["ids_test"], ids_test)

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)
