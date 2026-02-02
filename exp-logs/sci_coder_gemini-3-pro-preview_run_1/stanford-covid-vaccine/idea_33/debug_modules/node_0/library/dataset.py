import os
import torch
import pandas as pd
from library.config import Config, process_data, RNADataset


def get_dataset(
    mode: str,
    config: Config,
    load_cached_data: bool = True,
    debug: bool = False,
    num_samples: int = None,
) -> RNADataset:
    """
    Loads, processes, and caches the RNA dataset.

    This function manages the data pipeline:
    1. Checks for a cached processed file.
    2. If not found or forced reload, reads the raw Parquet file.
    3. Supports debugging by subsetting the data (if debug=True or num_samples is set).
    4. Processes the dataframe into tensors using library.config.process_data.
    5. Caches the processed data for future runs.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
        config (Config): Configuration object containing file paths and settings.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug (bool): If True, enables debug mode (loads a small subset).
        num_samples (int, optional): Specific number of samples to load. Overrides debug default.

    Returns:
        RNADataset: An instance of the RNA dataset ready for the DataLoader.
    """
    # Ensure the working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Determine cache filename
    # Use a distinct cache file for debug runs to avoid overwriting the full dataset cache
    if debug or num_samples is not None:
        cache_filename = f"{mode}_data_debug.pt"
    else:
        cache_filename = f"{mode}_data.pt"

    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        try:
            data_dict = torch.load(cache_path)
            return RNADataset(data_dict, mode=mode)
        except Exception as e:
            print(f"Error loading cache: {e}. Proceeding to re-process data.")

    # 2. Process from scratch
    print(f"Processing {mode} data (debug={debug}, num_samples={num_samples})...")

    # Identify the correct source file based on mode
    if mode == "train":
        parquet_path = config.TRAIN_PARQUET
    elif mode == "val":
        parquet_path = config.VAL_PARQUET
    elif mode == "test":
        parquet_path = config.TEST_PARQUET
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")

    # Load the raw data
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Handle Debugging / Subsetting
    if debug and num_samples is None:
        num_samples = 100  # Default debug size

    if num_samples is not None:
        print(f"Subsetting dataframe to first {num_samples} samples.")
        df = df.head(num_samples)

    # Process the dataframe into tensors
    # process_data handles structure parsing, distance calculation, and target filtering
    data_dict = process_data(df, mode=mode)

    # Save to cache
    print(f"Saving processed {mode} data to {cache_path}")
    torch.save(data_dict, cache_path)

    return RNADataset(data_dict, mode=mode)
