import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import contextlib
import ast
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer(contextlib.ContextDecorator):
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Completed in {elapsed:.6f} seconds.")


def safe_convert_list(x):
    """
    Robustly converts input to a list.
    Handles numpy arrays, python lists, and string representations of lists.
    Useful for the 'requester_subreddits_at_request' column.
    """
    if x is None:
        return []

    # If it's already a list or numpy array, convert to list
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)

    # If it's a string, try to parse it
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                # Fallback: treat as single item list or comma separated
                pass
        return [x]

    # Fallback for other types
    return [x]


def load_data(split="train", drop_leakage=True):
    """
    Loads the dataset for the specified split from the metadata parquet files.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        drop_leakage (bool): If True, removes columns defined in config.DROP_SUFFIXES
                             from train/val sets to prevent data leakage.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = config.TRAIN_DATA_PATH
    elif split == "val":
        path = config.VAL_DATA_PATH
    elif split == "test":
        path = config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_parquet(path)

    # Leakage prevention: Drop columns that contain future information (retrieval time stats)
    # These are present in train/val but should not be used for prediction.
    if drop_leakage and split in ["train", "val"]:
        leakage_cols = [
            col
            for col in df.columns
            if any(col.endswith(suffix) for suffix in config.DROP_SUFFIXES)
        ]

        if leakage_cols:
            # Filter out columns that are not in the dataframe (safety check)
            leakage_cols = [c for c in leakage_cols if c in df.columns]
            if leakage_cols:
                print(f"[{split}] Dropping {len(leakage_cols)} leakage columns.")
                df = df.drop(columns=leakage_cols)

    return df
