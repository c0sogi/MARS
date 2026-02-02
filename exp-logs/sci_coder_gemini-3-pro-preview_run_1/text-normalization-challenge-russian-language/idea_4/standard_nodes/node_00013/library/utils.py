import os
import json
import hashlib
import time
import pandas as pd
import numpy as np
import torch
from library.config import seed_everything


def get_unique_hash(config_dict=None, include_timestamp=False):
    """
    Generates a unique MD5 hash based on a configuration dictionary and optionally a timestamp.
    Used for artifact versioning.
    """
    hash_input = {}
    if config_dict:
        # Convert config object to dict if necessary, though dict is expected
        hash_input.update(config_dict)

    if include_timestamp:
        hash_input["timestamp"] = time.time()

    # Sort keys to ensure deterministic hashing of the dictionary
    # default=str handles non-serializable types by converting them to string
    s = json.dumps(hash_input, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def ensure_dir(file_path):
    """
    Ensures that the directory for the given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_metadata(path):
    """
    Loads a metadata CSV file (train/val/test) with appropriate data types.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    # Use object dtype for text columns to prevent pandas from misinterpreting tokens
    return pd.read_csv(path, dtype={"before": object, "after": object, "class": object})


def get_or_compute(cache_path, compute_func, load_cached_data=True, **kwargs):
    """
    Implements the strict caching logic flow:
    1. IF load_cached_data is True: Try to load the file.
    2. IF loading fails OR load_cached_data is False:
       - Compute/process the data using compute_func.
       - Save the result to cache_path.
    3. Return the data.

    Supports .parquet (pandas), .npy (numpy), and .pt/.pth (torch).
    """
    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=True)
            elif cache_path.endswith(".pt") or cache_path.endswith(".pth"):
                return torch.load(cache_path)
        except Exception as e:
            # If loading fails, proceed to recompute
            print(
                f"Warning: Failed to load cache from {cache_path} ({e}). Recomputing..."
            )

    # 2. Compute
    data = compute_func(**kwargs)

    # 3. Save
    ensure_dir(cache_path)
    if cache_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=False)
        else:
            raise ValueError("Data must be a DataFrame for .parquet extension")
    elif cache_path.endswith(".npy"):
        np.save(cache_path, data)
    elif cache_path.endswith(".pt") or cache_path.endswith(".pth"):
        torch.save(data, cache_path)

    return data


def save_submission(df, path):
    """
    Saves the submission DataFrame to CSV in the required format.
    Checks for 'id' and 'after' columns.
    """
    ensure_dir(path)
    if "id" not in df.columns or "after" not in df.columns:
        raise ValueError("Submission DataFrame must contain 'id' and 'after' columns.")

    # Ensure strict quoting or formatting if necessary, but standard CSV is usually fine
    # The sample submission uses standard CSV format.
    df[["id", "after"]].to_csv(path, index=False)


def print_metrics(metrics):
    """
    Prints validation metrics with full precision (no rounding).
    """
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
