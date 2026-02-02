import os
import pandas as pd
import ase.io
import library.config as config


def read_geometry(rel_path):
    """
    Reads an XYZ file from the input directory and returns an ASE Atoms object.

    Args:
        rel_path (str): Relative path to the geometry file (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The crystal structure object.
    """
    full_path = os.path.join(config.INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    # ASE read handles .xyz format automatically
    # Cite debug_lesson_3: Explicitly Specify File Formats When Extensions Are Misleading
    return ase.io.read(full_path, format="aims")


def load_metadata(split, load_cached_data=True):
    """
    Loads the metadata DataFrame for a specific split (train, val, test).
    Implements caching using Parquet files to persist the data (including debug sampling).

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from the cache first.

    Returns:
        pd.DataFrame: The metadata DataFrame.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Construct cache filename based on split and debug mode to avoid collisions
    cache_filename = f"{split}_metadata"
    if config.DEBUG:
        cache_filename += f"_debug_{config.DEBUG_SAMPLE_SIZE}"
    cache_filename += ".parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} metadata from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing from source.")

    # 2. Load from source CSVs
    print(f"Loading {split} metadata from source CSV...")
    if split == "train":
        source_path = config.TRAIN_METADATA_PATH
    elif split == "val":
        source_path = config.VAL_METADATA_PATH
    elif split == "test":
        source_path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata CSV not found at: {source_path}")

    df = pd.read_csv(source_path)

    # Apply Debug Sampling if enabled
    if config.DEBUG:
        print(
            f"DEBUG mode enabled: Sampling {config.DEBUG_SAMPLE_SIZE} rows from {split} set."
        )
        if len(df) > config.DEBUG_SAMPLE_SIZE:
            df = df.sample(
                n=config.DEBUG_SAMPLE_SIZE, random_state=config.RANDOM_SEED
            ).reset_index(drop=True)

    # 3. Save to cache
    print(f"Saving {split} metadata to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def get_data_generator(df):
    """
    Generator that yields metadata and the corresponding ASE Atoms object for each row.
    This ensures that every metadata entry is paired with its crystal structure.

    Args:
        df (pd.DataFrame): The metadata DataFrame containing a 'file_path' column.

    Yields:
        tuple: (index, row, atoms)
            - index: The index of the row in the DataFrame.
            - row: The pandas Series for the row.
            - atoms: The ase.Atoms object loaded from the file_path.
    """
    for idx, row in df.iterrows():
        atoms = read_geometry(row["file_path"])
        yield idx, row, atoms
