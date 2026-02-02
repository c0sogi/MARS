import os
import numpy as np
import pandas as pd
from library import config


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def load_metadata(split="train"):
    """
    Loads the train, validation, or test metadata from the ./metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = config.TRAIN_METADATA
    elif split == "val":
        path = config.VAL_METADATA
    elif split == "test":
        path = config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    print(f"Loading {split} metadata from {path}...")
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    return df


def load_structures(load_cached_data=True):
    """
    Loads molecular structures. Uses caching to speed up subsequent loads.
    Adds 'atomic_number' based on 'atom' symbol.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing structure info (molecule_name, atom_index, atom, x, y, z, atomic_number).
    """
    cache_path = os.path.join(config.CACHE_DIR, "structures_processed.parquet")

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading structures from cache: {cache_path}")
        try:
            df_structures = pd.read_parquet(cache_path)
            return df_structures
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    print(f"Processing structures from {config.STRUCTURES_CSV}...")
    df_structures = pd.read_csv(config.STRUCTURES_CSV)

    # Map atom symbol to atomic number
    # config.ATOM_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    df_structures["atomic_number"] = df_structures["atom"].map(config.ATOM_NUMBERS)

    # Optimize memory
    df_structures = reduce_mem_usage(df_structures)

    # Save to cache
    print(f"Saving processed structures to cache: {cache_path}")
    df_structures.to_parquet(cache_path, index=False)

    return df_structures
