import os
import random
import time
import numpy as np
import torch
import pandas as pd
from contextlib import contextmanager
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can reduce performance but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """
    Ensures that the directory for a given path exists.
    If path is a file path, ensures the parent directory exists.
    If path is a directory (ends with separator or no extension), ensures it exists.
    """
    # If it looks like a file (has an extension), get dirname.
    # Otherwise assume it is a directory.
    if os.path.splitext(path)[1]:
        dir_path = os.path.dirname(path)
    else:
        dir_path = path

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def print_header(title: str) -> None:
    """
    Prints a formatted header to stdout.
    """
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


@contextmanager
def timer(name: str):
    """
    Context manager to measure and print the execution time of a code block.
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"[{name}] done in {elapsed:.2f} s")


def save_to_cache(data, path: str) -> None:
    """
    Saves data to cache using parquet or npy formats.
    Enforces the requirement: Do NOT use pickle.
    """
    ensure_dir(path)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame for .parquet extension.")
    elif path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            raise ValueError("Data must be a numpy array for .npy extension.")
    else:
        raise ValueError(
            f"Unsupported file extension for caching: {path}. Use .parquet or .npy."
        )


def load_from_cache(path: str):
    """
    Loads data from cache if it exists. Returns None if not found.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(
            f"Unsupported file extension for caching: {path}. Use .parquet or .npy."
        )
