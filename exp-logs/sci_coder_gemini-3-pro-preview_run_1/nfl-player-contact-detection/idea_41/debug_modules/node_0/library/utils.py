import os
import sys
import logging
import random
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import CACHE_DIR, SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, Numpy, and OS environments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def setup_logger(log_file_path):
    """
    Configures and returns a logger that writes to both a file and standard output.
    Existing handlers are cleared to prevent duplicate logs.
    """
    logger = logging.getLogger("pipeline_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File Handler
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class CacheManager:
    """
    Manages the persistence of intermediate data structures (Parquet and Numpy files)
    to the configured cache directory.
    """

    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    def save_parquet(self, df, filename):
        """Saves a pandas DataFrame to a parquet file."""
        path = self._get_path(filename)
        df.to_parquet(path, index=False, engine="pyarrow")

    def load_parquet(self, filename):
        """Loads a pandas DataFrame from a parquet file if it exists."""
        path = self._get_path(filename)
        if os.path.exists(path):
            return pd.read_parquet(path, engine="pyarrow")
        return None

    def save_numpy(self, arr, filename):
        """Saves a numpy array to a .npy file."""
        path = self._get_path(filename)
        np.save(path, arr)

    def load_numpy(self, filename):
        """Loads a numpy array from a .npy file if it exists."""
        path = self._get_path(filename)
        if os.path.exists(path):
            return np.load(path)
        return None

    def exists(self, filename):
        """Checks if a file exists in the cache directory."""
        return os.path.exists(self._get_path(filename))


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between ground truth and binary predictions.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, steps=100):
    """
    Iterates through probability thresholds to find the one that maximizes the MCC score.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities.
        steps (int): Number of threshold steps to evaluate between 0.01 and 0.99.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_mcc = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_binary)

        if score > best_mcc:
            best_mcc = score
            best_thresh = thresh

    return best_thresh, best_mcc
