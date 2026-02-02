import os
import numpy as np
import pandas as pd
from library.config import (
    METADATA_DIR,
    CACHE_DIR,
    FLOAT_PRECISION,
    RANDOM_SEED,
)
from library.feature_engineering import get_macro_features


def load_metadata(split):
    """
    Loads the metadata CSV for the given split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe containing IDs, labels (if available),
                      and global features.
    """
    path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)
    return df


def get_combined_dataset(split, load_cached_data=True, sample_size=None):
    """
    Constructs the combined dataset consisting of the 192 Global features
    and the 11 Macro features. Handles caching and precision casting.

    Structure of returned X:
        - Columns 0-191: Global Features (Margin, Shape, Texture)
        - Columns 192-202: Macro Features (Hu Moments, Geometric Properties)

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load pre-computed numpy arrays.
        sample_size (int, optional): If provided, limits the dataset size for debugging.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): Feature matrix of shape (N, 203) in float64.
            y (np.ndarray): Target labels of shape (N,) or None for test.
            ids (np.ndarray): Image IDs of shape (N,).
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_X_path = os.path.join(CACHE_DIR, f"X_{split}_combined.npy")
    cache_y_path = os.path.join(CACHE_DIR, f"y_{split}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"ids_{split}.npy")

    # Attempt to load from cache
    if load_cached_data:
        has_X = os.path.exists(cache_X_path)
        has_ids = os.path.exists(cache_ids_path)
        has_y = os.path.exists(cache_y_path)

        # For test split, y file is not expected
        if has_X and has_ids:
            if split != "test" and not has_y:
                # Cache incomplete for supervised set, proceed to compute
                pass
            else:
                print(f"Loading cached combined dataset for '{split}'...")
                X = np.load(cache_X_path)
                ids = np.load(cache_ids_path)
                y = np.load(cache_y_path) if has_y else None

                # Apply sampling if requested
                if sample_size is not None and len(X) > sample_size:
                    return (
                        X[:sample_size],
                        (y[:sample_size] if y is not None else None),
                        ids[:sample_size],
                    )
                return X, y, ids

    print(f"Computing combined dataset for '{split}' (Cache miss or force reload)...")

    # 1. Load Metadata (contains Global features + Labels + IDs)
    df_meta = load_metadata(split)

    # 2. Load Macro Features (computed via feature_engineering)
    # We pass the same load_cached_data flag down to leverage feature-level caching
    df_macro = get_macro_features(split, load_cached_data=load_cached_data)

    # 3. Merge Datasets
    # Both dataframes must have 'id' column
    # Inner join ensures we only keep rows present in both
    df_combined = pd.merge(df_meta, df_macro, on="id", how="inner")

    # Verify merge consistency
    if len(df_combined) != len(df_meta):
        raise ValueError(
            f"Merge inconsistency: Metadata has {len(df_meta)} rows, "
            f"Combined has {len(df_combined)} rows."
        )

    # 4. Extract Features (X)
    # Identify global columns (192 columns) by prefix
    global_cols = [
        c for c in df_combined.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Identify macro columns (everything in df_macro except 'id')
    macro_cols = [c for c in df_macro.columns if c != "id"]

    # Concatenate column lists to ensure deterministic order: Global then Macro
    feature_cols = global_cols + macro_cols

    # Cast to strictly defined float precision
    X = df_combined[feature_cols].values.astype(FLOAT_PRECISION)
    ids = df_combined["id"].values

    # 5. Extract Targets (y)
    if "species" in df_combined.columns:
        y = df_combined["species"].values
    else:
        y = None

    # 6. Save to Cache
    print(f"Saving combined dataset for '{split}' to {CACHE_DIR}...")
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    # 7. Apply sampling if requested
    if sample_size is not None and len(X) > sample_size:
        return (
            X[:sample_size],
            (y[:sample_size] if y is not None else None),
            ids[:sample_size],
        )

    return X, y, ids
