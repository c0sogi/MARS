import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_artifact, save_artifact, set_seed


def load_raw_metadata():
    """
    Loads the raw train, validation, and test metadata files from the
    directory specified in Config.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"Loading raw metadata from {Config.METADATA_DIR}...")

    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_PATH}")
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(f"Val metadata not found at {Config.VAL_PATH}")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_PATH}")

    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    return train_df, val_df, test_df


def get_data(load_cached_data: bool = True):
    """
    Retrieves the Union Training Dataset (Train + Val) and the Test Dataset.
    Implements caching to avoid re-loading/re-processing raw files.

    If Config.DEBUG is True, returns a small subset of the data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed
                                 parquets from the working directory.

    Returns:
        tuple: (union_train_df, test_df)
    """
    set_seed(Config.RANDOM_SEED)

    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "union_train.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
            print(f"Loading data from cache: {Config.WORKING_DIR}")
            try:
                union_train_df = load_artifact(cache_train_path)
                test_df = load_artifact(cache_test_path)
                return union_train_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")
        else:
            print("Cache not found. Processing from source.")
    else:
        print("Skipping cache. Processing from source.")

    # 2. Load raw metadata
    train_df, val_df, test_df = load_raw_metadata()

    # 3. Create Union Dataset (Train + Val)
    # The strategy requires merging provided train and val splits into a single
    # dataset for cross-validation within the pipeline.
    print("Creating Union Dataset (Train + Val)...")
    union_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # 4. Handle Debug Mode
    if Config.DEBUG:
        print("DEBUG Mode enabled: Downsampling data to 100 rows.")
        union_train_df = union_train_df.head(100).copy()
        test_df = test_df.head(100).copy()

    # 5. Save to cache
    print(f"Saving processed data to cache: {Config.WORKING_DIR}")
    save_artifact(union_train_df, cache_train_path)
    save_artifact(test_df, cache_test_path)

    print(
        f"Data loaded. Union Train shape: {union_train_df.shape}, Test shape: {test_df.shape}"
    )
    return union_train_df, test_df
