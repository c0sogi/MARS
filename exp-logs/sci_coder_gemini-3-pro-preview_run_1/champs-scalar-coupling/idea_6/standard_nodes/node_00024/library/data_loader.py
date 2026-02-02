import os
import pandas as pd
import numpy as np
from library import config


def _downcast_dtypes(df):
    """
    Downcasts numeric columns in a DataFrame to save memory.
    float64 -> float32
    int64 -> int32/int16/int8
    objects -> category (if low cardinality)
    """
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_object_dtype(col_type):
            # Check for low cardinality to convert to category
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")

    return df


def load_metadata(split="train"):
    """
    Loads the metadata for a specific data split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata DataFrame with downcasted types.
    """
    if split == "train":
        path = config.TRAIN_METADATA_PATH
    elif split == "val":
        path = config.VAL_METADATA_PATH
    elif split == "test":
        path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    print(f"Loading {split} metadata from {path}...")
    df = pd.read_csv(path)
    df = _downcast_dtypes(df)

    # Ensure atom indices are integers (sometimes read as float if NaNs exist, though unlikely here)
    if "atom_index_0" in df.columns:
        df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    if "atom_index_1" in df.columns:
        df["atom_index_1"] = df["atom_index_1"].astype(np.int32)

    return df


def load_structures(load_cached_data=True):
    """
    Loads the molecular structures data. Implements caching to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load from a local parquet cache.
                                 If False or cache missing, loads from raw CSV and updates cache.

    Returns:
        pd.DataFrame: DataFrame containing ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'].
    """
    cache_path = os.path.join(config.WORKING_DIR, "structures_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading structures from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing raw data.")

    # 2. Load from raw CSV
    raw_path = config.STRUCTURES_PATH
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw structures file not found: {raw_path}")

    print(f"Loading raw structures from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 3. Rename columns if necessary (structures.csv usually has standard headers)
    # Expected: molecule_name, atom_index, atom, x, y, z
    # If raw file differs, we map it here. Based on description, it matches.

    # 4. Optimize memory
    print("Optimizing memory usage for structures...")
    df = _downcast_dtypes(df)

    # Ensure atom_index is int32
    if "atom_index" in df.columns:
        df["atom_index"] = df["atom_index"].astype(np.int32)

    # 5. Save to cache
    print(f"Saving processed structures to cache: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
