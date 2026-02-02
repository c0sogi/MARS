import os
import pandas as pd
from library import config
from library.utils import Timer


def load_datasets(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.

    Combines the 'train' and 'val' splits from the metadata directory into a single
    DataFrame to allow for flexible Cross-Validation strategies (e.g., 5-fold)
    and full-dataset retraining in the downstream pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load the pre-concatenated
                                 training data from the working directory.

    Returns:
        tuple: (df_train, df_test)
            - df_train: Combined training and validation data.
            - df_test: Test data.
    """

    # Define paths
    metadata_train_path = os.path.join(config.METADATA_DIR, "train.parquet")
    metadata_val_path = os.path.join(config.METADATA_DIR, "val.parquet")
    metadata_test_path = os.path.join(config.METADATA_DIR, "test.parquet")

    # Cache path for the combined training set
    cached_train_path = os.path.join(config.WORKING_DIR, "combined_train.parquet")

    df_train = None
    df_test = None

    with Timer("Loading Datasets"):
        # ---------------------------------------------------------
        # 1. Load Test Data (Always load from metadata source)
        # ---------------------------------------------------------
        if os.path.exists(metadata_test_path):
            df_test = pd.read_parquet(metadata_test_path)
        else:
            raise FileNotFoundError(f"Test metadata not found at {metadata_test_path}")

        # ---------------------------------------------------------
        # 2. Load Training Data (Cached or Source)
        # ---------------------------------------------------------
        # Attempt to load from cache if requested
        if load_cached_data and os.path.exists(cached_train_path):
            try:
                print(f"Loading cached training data from {cached_train_path}...")
                df_train = pd.read_parquet(cached_train_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                df_train = None

        # If cache missing or load failed, process from scratch
        if df_train is None:
            print("Loading raw metadata and concatenating train/val splits...")

            if not os.path.exists(metadata_train_path):
                raise FileNotFoundError(
                    f"Train metadata not found at {metadata_train_path}"
                )
            if not os.path.exists(metadata_val_path):
                raise FileNotFoundError(
                    f"Val metadata not found at {metadata_val_path}"
                )

            df_train_part = pd.read_parquet(metadata_train_path)
            df_val_part = pd.read_parquet(metadata_val_path)

            # Concatenate train and val to form the full training set
            # We use ignore_index=True to reset the index after concatenation
            df_train = pd.concat(
                [df_train_part, df_val_part], axis=0, ignore_index=True
            )

            # Save to cache
            print(f"Saving combined training data to {cached_train_path}...")
            df_train.to_parquet(cached_train_path, index=False)

    print(f"Data Loaded: Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return df_train, df_test
