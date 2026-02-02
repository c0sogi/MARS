import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    TARGET_COLS,
)
from library.features import generate_features


def load_data(split_name: str, load_cached_data: bool = True):
    """
    Loads and processes data for a specific split (train, val, test).

    This function orchestrates the data loading pipeline:
    1. Checks for cached processed data (X and y matrices).
    2. If not found or forced reload, loads raw metadata.
    3. Calls `generate_features` to compute geometric descriptors.
    4. Merges geometric descriptors with tabular metadata.
    5. Applies log1p transformation to target variables for train/val splits.
    6. Caches the final processed DataFrames to disk.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target DataFrame (or None for test).
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths for the final processed matrices
    cache_X_path = os.path.join(WORKING_DIR, f"{split_name}_X.parquet")
    cache_y_path = os.path.join(WORKING_DIR, f"{split_name}_y.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        # Check if X exists
        if os.path.exists(cache_X_path):
            # For test set, we only need X
            if split_name == "test":
                print(f"Loading cached {split_name} features from {cache_X_path}")
                X = pd.read_parquet(cache_X_path)
                return X, None
            # For train/val, we need both X and y
            elif os.path.exists(cache_y_path):
                print(
                    f"Loading cached {split_name} data from {cache_X_path} and {cache_y_path}"
                )
                X = pd.read_parquet(cache_X_path)
                y = pd.read_parquet(cache_y_path)
                return X, y

    # 2. Compute from scratch
    print(f"Processing {split_name} data from scratch...")

    # Identify metadata file path
    if split_name == "train":
        metadata_path = TRAIN_METADATA_PATH
    elif split_name == "val":
        metadata_path = VAL_METADATA_PATH
    elif split_name == "test":
        metadata_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Generate Geometric Features
    # This function handles its own caching for the expensive part (atom processing)
    # We pass load_cached_data to it as well to respect the flag.
    df_geo = generate_features(df_meta, split_name, load_cached_data=load_cached_data)

    # Merge Tabular Metadata
    # We keep the tabular features provided in the CSV (e.g., lattice vectors, angles, composition)
    # We exclude ID, file_path, and targets from the feature matrix X
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    tabular_cols = [c for c in df_meta.columns if c not in exclude_cols]

    # Concatenate tabular features with geometric features
    # Pandas aligns by index automatically, which is preserved from df_meta
    X = pd.concat([df_meta[tabular_cols], df_geo], axis=1)

    # Handle Targets
    y = None
    if split_name in ["train", "val"]:
        # Extract targets
        y_raw = df_meta[TARGET_COLS]
        # Log transformation: z = log(1 + y)
        # This helps with the skewed distribution and strictly positive nature of energies
        y = np.log1p(y_raw)

        # Save y to cache
        print(f"Saving {split_name} targets to {cache_y_path}")
        y.to_parquet(cache_y_path)

    # Save X to cache
    print(f"Saving {split_name} features to {cache_X_path}")
    X.to_parquet(cache_X_path)

    return X, y


def inverse_transform_targets(y_pred):
    """
    Applies inverse transformation to predictions: exp(z) - 1.
    Used for generating final submission values from model outputs.

    Args:
        y_pred (np.array or pd.DataFrame): Log-transformed predictions.

    Returns:
        np.array or pd.DataFrame: Predictions in original scale.
    """
    return np.expm1(y_pred)
