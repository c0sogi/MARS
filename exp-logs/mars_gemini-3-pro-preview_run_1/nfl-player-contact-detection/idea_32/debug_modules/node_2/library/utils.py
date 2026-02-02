import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from functools import wraps
from library.config import Config


def setup_logging(level=logging.INFO):
    """
    Configures the logging system for the application.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def parameter_aware_cache(cache_path, file_format="parquet"):
    """
    Decorator to cache function results to disk using Parquet or NPY formats.

    Logic:
    1. Checks if `load_cached_data` is True in kwargs.
    2. If True and file exists, loads from disk.
    3. Otherwise, runs the function, saves to disk, and returns result.

    Args:
        cache_path (str): The absolute path where the file should be saved/loaded.
        file_format (str): 'parquet' for pandas DataFrames, 'npy' for numpy arrays.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            load_cached_data = kwargs.get("load_cached_data", False)

            # Try to load from cache
            if load_cached_data:
                if os.path.exists(cache_path):
                    logging.info(f"Loading cached data from {cache_path}")
                    try:
                        if file_format == "parquet":
                            return pd.read_parquet(cache_path)
                        elif file_format == "npy":
                            return np.load(cache_path)
                        else:
                            raise ValueError(f"Unsupported file format: {file_format}")
                    except Exception as e:
                        logging.warning(
                            f"Failed to load cache from {cache_path}: {e}. Recomputing..."
                        )
                else:
                    logging.info(
                        f"Cache file not found at {cache_path}. Recomputing..."
                    )

            # Compute result
            result = func(*args, **kwargs)

            # Save to cache
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                logging.info(f"Saving data to cache at {cache_path}")
                if file_format == "parquet":
                    if isinstance(result, pd.DataFrame):
                        result.to_parquet(cache_path, index=False)
                    else:
                        logging.error(
                            "Result is not a DataFrame, cannot save as parquet."
                        )
                elif file_format == "npy":
                    if isinstance(result, np.ndarray):
                        np.save(cache_path, result)
                    else:
                        logging.error(
                            "Result is not a numpy array, cannot save as npy."
                        )
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
            except Exception as e:
                logging.error(f"Failed to save cache to {cache_path}: {e}")

            return result

        return wrapper

    return decorator
