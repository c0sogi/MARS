import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    FLOAT_PRECISION,
)
from library.image_processing import extract_morphometric_features


def load_dataset(split_name: str, load_cached_data: bool = True) -> dict:
    """
    Loads the dataset for a specific split (train, val, or test), separating
    Global features, Morphometric features, and targets. Implements caching
    for deterministic data processing.

    Args:
        split_name (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing:
            - 'global_view': np.ndarray of shape (N, 192)
            - 'morph_view': np.ndarray of shape (N, 11)
            - 'y': np.ndarray of shape (N,) or None
            - 'ids': np.ndarray of shape (N,)
    """
    # 1. Resolve Metadata Path
    if split_name == "train":
        data_path = TRAIN_DATA_PATH
    elif split_name == "val":
        data_path = VAL_DATA_PATH
    elif split_name == "test":
        data_path = TEST_DATA_PATH
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Metadata file not found: {data_path}")

    # 2. Load Metadata DataFrame
    # We load this regardless of cache status because extract_morphometric_features
    # requires the dataframe to function (even if it hits its own cache).
    # CSV loading is fast enough to not require caching itself.
    df = pd.read_csv(data_path)

    # 3. Extract Morphometric Features
    # This function manages its own caching in WORKING_DIR/morphometrics_{split_name}.npy
    X_morph = extract_morphometric_features(
        df, dataset_name=split_name, load_cached_data=load_cached_data
    )

    # 4. Handle Global Features, Targets, and IDs with Caching
    # Define cache paths
    cache_global_path = os.path.join(WORKING_DIR, f"data_{split_name}_X_global.npy")
    cache_y_path = os.path.join(WORKING_DIR, f"data_{split_name}_y.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"data_{split_name}_ids.npy")

    # Determine if we can load from cache
    # We need all relevant caches to exist. For test, y cache might not exist or be needed.
    # We handle y separately.
    cache_exists = (
        os.path.exists(cache_global_path)
        and os.path.exists(cache_ids_path)
        and (os.path.exists(cache_y_path) if "species" in df.columns else True)
    )

    if load_cached_data and cache_exists:
        # --- Cache Hit ---
        X_global = np.load(cache_global_path)
        ids = np.load(cache_ids_path)

        if os.path.exists(cache_y_path):
            y = np.load(cache_y_path, allow_pickle=True)
        else:
            y = None

    else:
        # --- Cache Miss or Forced Reload ---

        # Extract Global Features (192 columns)
        # Columns start with margin, shape, or texture
        feature_cols = [
            c
            for c in df.columns
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]
        # Sort to ensure deterministic order (though usually they are ordered in CSV)
        # We rely on CSV order but filtering ensures we only get features.
        # The prompt implies the CSV has these columns.

        X_global = df[feature_cols].values.astype(FLOAT_PRECISION)

        # Extract IDs
        ids = df["id"].values

        # Extract Targets
        if "species" in df.columns:
            y = df["species"].values
            # Save y
            np.save(cache_y_path, y)
        else:
            y = None
            # If a stale y cache exists but current df has no species (unlikely for same split name),
            # we don't overwrite/delete, just don't save.

        # Save Global and IDs
        np.save(cache_global_path, X_global)
        np.save(cache_ids_path, ids)

    # 5. Return Data Dictionary
    return {"global_view": X_global, "morph_view": X_morph, "y": y, "ids": ids}
