import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from library.config import WORKING_DIR, SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_filename="execution.log", log_dir=WORKING_DIR):
    """
    Configures logging to output to both a file and standard output.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Remove existing handlers to avoid duplication if called multiple times
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


class CacheManager:
    """
    Manages deterministic data processing with caching.
    Enforces strict checks on loading/saving intermediate files (Parquet/Numpy)
    to avoid re-computation and ensure consistency.
    """

    def __init__(self, cache_dir=WORKING_DIR, load_cached_data=True):
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    def load(self, filename):
        """
        Attempts to load data from the cache directory.
        Returns None if load_cached_data is False, file is missing, or format is unsupported.
        """
        if not self.load_cached_data:
            return None

        filepath = self._get_path(filename)
        if not os.path.exists(filepath):
            return None

        try:
            if filename.endswith(".parquet"):
                return pd.read_parquet(filepath)
            elif filename.endswith(".npy"):
                return np.load(filepath)
            else:
                print(
                    f"Warning: CacheManager received unsupported load format for {filename}. Only .parquet and .npy are supported."
                )
                return None
        except Exception as e:
            print(f"Error loading cached file {filename}: {e}")
            return None

    def save(self, data, filename):
        """
        Saves data to the cache directory using Parquet (for DataFrames) or NPY (for Arrays).
        """
        filepath = self._get_path(filename)
        try:
            if filename.endswith(".parquet"):
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(filepath, index=False)
                else:
                    print(
                        f"Error: Expected pd.DataFrame for .parquet save, got {type(data)}."
                    )
            elif filename.endswith(".npy"):
                if isinstance(data, np.ndarray):
                    np.save(filepath, data)
                else:
                    print(
                        f"Error: Expected np.ndarray for .npy save, got {type(data)}."
                    )
            else:
                print(
                    f"Warning: CacheManager received unsupported save format for {filename}. Only .parquet and .npy are supported."
                )
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

    def execute_with_cache(self, filename, func, *args, **kwargs):
        """
        Executes a processing function with caching logic.

        Logic:
        1. IF load_cached_data is True AND file exists: Load and return.
        2. ELSE: Execute func(*args, **kwargs), Save result, Return result.
        """
        # 1. Try to load
        cached_data = self.load(filename)
        if cached_data is not None:
            print(f"Cache hit: Loaded {filename}")
            return cached_data

        # 2. Compute
        print(f"Cache miss: Computing {filename}...")
        result = func(*args, **kwargs)

        # 3. Save
        if result is not None:
            self.save(result, filename)
            print(f"Saved computed data to {filename}")

        return result
