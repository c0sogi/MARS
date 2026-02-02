import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Union, Optional, Any, List
from library.config import Paths


def setup_logger(
    name: str = "main", log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, defaults to 'train.log' in WORKING_DIR.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file is None:
        log_file = os.path.join(Paths.WORKING_DIR, "train.log")

    # Ensure directory exists for log file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_category_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    # float16 has lower precision, sticking to float32 for safety in ML
                    df[col] = df[col].astype(np.float32)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(
            f"Memory usage optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


class CacheManager:
    """
    Manages saving and loading of intermediate artifacts (Parquet, NPY, Models)
    to the configured working directory.
    """

    def __init__(self, base_dir: Path = Paths.WORKING_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_path(self, filename: str) -> Path:
        return self.base_dir / filename

    def exists(self, filename: str) -> bool:
        return self.get_path(filename).exists()

    def save_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Saves a DataFrame to parquet."""
        path = self.get_path(filename)
        # Ensure index is reset or handled if needed, but usually direct save is fine
        df.to_parquet(path, index=False)
        print(f"Saved DataFrame to {path}")

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Loads a DataFrame from parquet."""
        path = self.get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        print(f"Loading DataFrame from {path}")
        return pd.read_parquet(path)

    def save_npy(self, array: np.ndarray, filename: str) -> None:
        """Saves a NumPy array."""
        path = self.get_path(filename)
        np.save(path, array)
        print(f"Saved NumPy array to {path}")

    def load_npy(self, filename: str) -> np.ndarray:
        """Loads a NumPy array."""
        path = self.get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        print(f"Loading NumPy array from {path}")
        return np.load(path)

    def save_torch_model(self, model: torch.nn.Module, filename: str) -> None:
        """Saves a PyTorch model state dict."""
        path = self.get_path(filename)
        torch.save(model.state_dict(), path)
        print(f"Saved PyTorch model to {path}")

    def load_torch_model(
        self, model: torch.nn.Module, filename: str, device: str = "cpu"
    ) -> torch.nn.Module:
        """Loads a PyTorch model state dict into the provided model instance."""
        path = self.get_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"Loading PyTorch model from {path}")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model
