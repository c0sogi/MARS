import os
import sys
import logging
import time
import json
import contextlib
import pandas as pd
import numpy as np
from library.config import Config, set_seed


def setup_logger(name="main", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger that logs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, uses a default in WORKING_DIR.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "execution.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class Timer(contextlib.ContextDecorator):
    """
    Context manager to measure and log execution time of a block.
    """

    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting: {self.name}")
        else:
            print(f"Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        msg = f"Finished: {self.name} in {duration:.4f} seconds"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def load_split_data(split, logger=None):
    """
    Loads the data for a specific split (train, val, test) by merging
    metadata with the raw JSON data.

    Args:
        split (str): One of 'train', 'val', 'test'.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        pd.DataFrame: The merged DataFrame containing features and labels (if available).
    """
    if logger:
        logger.info(f"Loading data for split: {split}")

    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_META
    elif split == "val":
        meta_path = Config.VAL_META
    elif split == "test":
        meta_path = Config.TEST_META
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Load Metadata
    df_meta = pd.read_csv(meta_path)

    # Identify source files (usually input/train.json or input/test.json)
    # We assume paths in metadata are relative to the project root
    source_files = df_meta["source_file"].unique()

    raw_data_map = {}
    for src in source_files:
        # Construct full path. Metadata contains 'input/train.json', Config.INPUT_DIR is './input'
        # We assume the metadata source_file is a relative path that can be found directly or relative to root
        if os.path.exists(src):
            full_path = src
        else:
            # Try joining with current directory if strictly relative
            full_path = os.path.join(".", src)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Source raw file not found: {full_path}")

        if full_path not in raw_data_map:
            if logger:
                logger.info(f"Loading raw JSON: {full_path}")
            with open(full_path, "r") as f:
                raw_data_map[src] = json.load(f)

    # Merge Data
    # We can use sample_index for fast lookup if we trust it, or merge on request_id
    # Merging on request_id is safer.

    # Convert raw data lists to DataFrames for merging
    raw_dfs = []
    for src, data_list in raw_data_map.items():
        raw_dfs.append(pd.DataFrame(data_list))

    if not raw_dfs:
        raise ValueError("No raw data loaded.")

    df_raw = pd.concat(raw_dfs, ignore_index=True)

    # Drop duplicates in raw data if any (e.g. if multiple splits point to same file)
    df_raw = df_raw.drop_duplicates(subset=["request_id"])

    # Merge metadata with raw data
    # Metadata controls the rows for this split
    df_merged = df_meta.merge(
        df_raw, on="request_id", how="left", suffixes=("", "_raw")
    )

    # Clean up potential duplicate columns from merge if raw had them
    to_drop = [c for c in df_merged.columns if c.endswith("_raw")]
    if to_drop:
        df_merged.drop(columns=to_drop, inplace=True)

    if logger:
        logger.info(f"Loaded {len(df_merged)} samples for split '{split}'")

    return df_merged


def get_or_compute(func, cache_path, load_cached_data=True, logger=None, **kwargs):
    """
    Caches the result of a function to a file (Parquet or NumPy).

    Args:
        func (callable): The function to compute the data if cache is missing.
        cache_path (str): Path where the result should be saved/loaded.
        load_cached_data (bool): If True, attempts to load from cache first.
        logger (logging.Logger, optional): Logger instance.
        **kwargs: Arguments passed to func.

    Returns:
        The data (pd.DataFrame or np.ndarray).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try to load
    if load_cached_data and os.path.exists(cache_path):
        if logger:
            logger.info(f"Loading cached data from {cache_path}")

        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path)
            else:
                raise ValueError("Unsupported cache file format. Use .parquet or .npy")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    if logger:
        logger.info(f"Computing data for {cache_path}...")

    result = func(**kwargs)

    # Save
    if logger:
        logger.info(f"Saving data to {cache_path}")

    if cache_path.endswith(".parquet"):
        if isinstance(result, pd.DataFrame):
            result.to_parquet(cache_path)
        else:
            raise TypeError("Result is not a DataFrame but cache path is .parquet")
    elif cache_path.endswith(".npy"):
        if isinstance(result, np.ndarray):
            np.save(cache_path, result)
        else:
            raise TypeError("Result is not a ndarray but cache path is .npy")
    else:
        raise ValueError("Unsupported cache file format. Use .parquet or .npy")

    return result


def print_metrics(metrics, logger=None):
    """
    Prints metrics dictionary with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        logger (logging.Logger, optional): Logger instance.
    """
    msg_lines = ["Validation Metrics:"]
    for k, v in metrics.items():
        msg_lines.append(f"  {k}: {v}")

    msg = "\n".join(msg_lines)

    if logger:
        logger.info(msg)
    else:
        print(msg)
