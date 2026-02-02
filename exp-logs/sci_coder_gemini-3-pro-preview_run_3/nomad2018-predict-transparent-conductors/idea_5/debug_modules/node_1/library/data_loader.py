import os
import pandas as pd
import ase.io
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    WORKING_DIR,
    RANDOM_SEED,
)


def load_metadata(split="train", max_samples=None, load_cached_data=True):
    """
    Loads metadata for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        max_samples (int, optional): If set, limits the number of rows loaded.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    # Determine source path
    if split == "train":
        csv_path = TRAIN_METADATA_PATH
    elif split == "val":
        csv_path = VAL_METADATA_PATH
    elif split == "test":
        csv_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Define cache path
    cache_path = os.path.join(WORKING_DIR, f"{split}_metadata.parquet")

    df = None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} metadata from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source if not loaded
    if df is None:
        print(f"Loading {split} metadata from source: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Save to cache
        try:
            df.to_parquet(cache_path, index=False)
            print(f"Saved {split} metadata to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # Apply max_samples limit if provided
    if max_samples is not None and max_samples < len(df):
        print(
            f"Limiting {split} dataset to {max_samples} samples (Random Seed: {RANDOM_SEED})."
        )
        df = df.sample(n=max_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Loaded {len(df)} rows for {split} split.")
    return df


def load_geometries(df):
    """
    Loads ASE Atoms objects for each row in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing a 'file_path' column.

    Returns:
        list: A list of ase.Atoms objects.
    """
    print(f"Loading geometries for {len(df)} samples...")
    atoms_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Geometry file not found: {full_path}")

        # Read the xyz file
        # format='xyz' is standard for these files
        atoms = ase.io.read(full_path, format="xyz")
        atoms_list.append(atoms)

    print(f"Successfully loaded {len(atoms_list)} geometry objects.")
    return atoms_list
