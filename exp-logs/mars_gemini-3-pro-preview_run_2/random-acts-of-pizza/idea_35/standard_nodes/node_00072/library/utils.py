import os
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Also configures CUDA for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Suppress warnings for cleaner output as requested
    warnings.filterwarnings("ignore")


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Configures a logger to output to both a file and the console.
    Ensures handlers are not duplicated if the logger is retrieved multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def compute_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.
    Returns the raw float value for full precision.
    """
    return roc_auc_score(y_true, y_pred)


def ensure_dir(path: str):
    """
    Ensures that the directory for a given path exists.
    """
    if os.path.splitext(path)[1]:  # It's a file path
        directory = os.path.dirname(path)
    else:  # It's a directory path
        directory = path

    if directory:
        os.makedirs(directory, exist_ok=True)


def save_to_cache(data, path: str):
    """
    Saves data to cache using approved formats (npy for numpy, parquet for pandas).
    Strictly avoids pickle.
    """
    ensure_dir(path)

    if path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            raise ValueError(
                f"Expected numpy array for .npy extension, got {type(data)}"
            )

    elif path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError(
                f"Expected pandas DataFrame for .parquet extension, got {type(data)}"
            )
    else:
        raise ValueError("Unsupported cache format. Use .npy or .parquet")


def load_from_cache(path: str):
    """
    Loads data from cache if it exists. Returns None if file is missing.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported cache format. Use .npy or .parquet")
