import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
    WORKING_DIR,
)
from library.features import process_dataset


def load_and_process_data(
    dataset_type="train", load_cached_data=True, max_samples=None
):
    """
    Loads metadata, processes features (using caching), and prepares X and y matrices.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load features from cache if available.
        max_samples (int, optional): If set, truncates the data to this number of samples for debugging.

    Returns:
        tuple:
            - If train/val: (X, y) where X is feature DataFrame and y is target DataFrame (log-transformed).
            - If test: (X, ids) where X is feature DataFrame and ids is the Series of IDs.
    """
    # 1. Identify Metadata Path
    if dataset_type == "train":
        meta_path = TRAIN_METADATA_PATH
    elif dataset_type == "val":
        meta_path = VAL_METADATA_PATH
    elif dataset_type == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # 2. Process Dataset (Feature Extraction + Caching)
    # This function from library.features handles reading XYZ files, computing RDF/Voronoi features,
    # and caching the result to parquet.
    df_features = process_dataset(
        metadata_df=pd.read_csv(meta_path),
        load_cached_data=load_cached_data,
        dataset_name=dataset_type,
    )

    # 3. Apply Subsampling (if requested)
    if max_samples is not None:
        print(f"Subsampling {dataset_type} data to {max_samples} samples.")
        df_features = df_features.head(max_samples)

    # 4. Prepare X (Features) and y (Targets)
    # Define columns to exclude from the feature matrix X
    # We exclude IDs, file paths, and the target columns themselves
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].copy()

    # Handle Targets based on dataset type
    if dataset_type in ["train", "val"]:
        # Extract targets
        y = df_features[TARGET_COLS].copy()

        # Log-transform targets: z = log(1 + y)
        # This stabilizes variance for energy values which are strictly positive
        y_log = np.log1p(y)

        print(f"Loaded {dataset_type} data: X.shape={X.shape}, y.shape={y_log.shape}")
        return X, y_log

    else:
        # For test set, we need IDs for submission matching
        ids = df_features["id"].copy()
        print(f"Loaded {dataset_type} data: X.shape={X.shape}, ids.shape={ids.shape}")
        return X, ids


def inverse_transform_targets(y_log_pred):
    """
    Applies the inverse transformation to predictions: y = exp(z) - 1.
    Used to convert model output back to original energy units (eV).

    Args:
        y_log_pred (np.array or pd.DataFrame): Log-transformed predictions.

    Returns:
        np.array or pd.DataFrame: Predictions in original scale.
    """
    return np.expm1(y_log_pred)
