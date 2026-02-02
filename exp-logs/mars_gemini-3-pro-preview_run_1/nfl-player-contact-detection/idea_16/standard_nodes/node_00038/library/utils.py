import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import PathConfig


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for the Dual-Scout ensemble training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def optimize_mcc_threshold(y_true, y_pred_proba):
    """
    Finds the optimal probability threshold to maximize Matthews Correlation Coefficient.

    Args:
        y_true: Array-like of true binary labels.
        y_pred_proba: Array-like of predicted probabilities.

    Returns:
        best_threshold: The threshold (0.01-0.99) that yields the highest MCC.
        best_mcc: The highest MCC value achieved.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    best_mcc = -1.0
    best_threshold = 0.5

    # Granular search space for precise threshold tuning
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        # Calculate MCC for current threshold
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    # Print full precision metrics
    print(f"Best Threshold: {best_threshold}")
    print(f"Best MCC: {best_mcc}")

    return best_threshold, best_mcc


class CacheManager:
    """
    Handles saving and loading of artifacts (Parquet/Numpy) to disk.
    Supports the iterative development pipeline by caching processed features
    and hard-negative indices to the working directory.
    """

    def __init__(self, cache_dir=PathConfig.WORKING_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _resolve_path(self, filename):
        """
        Resolves the full file path.
        If filename is absolute (from Config) or already contains the cache directory, uses it directly.
        Otherwise, joins with the default cache directory.
        """
        # Fix: Prevent double nesting of paths (Cite debug_lesson_20)
        if (
            os.path.isabs(filename)
            or filename.startswith(self.cache_dir)
            or filename.startswith("./")
        ):
            return filename
        return os.path.join(self.cache_dir, filename)

    def load_parquet(self, filename):
        """
        Attempts to load a DataFrame from a parquet file.
        Returns None if the file does not exist.
        """
        path = self._resolve_path(filename)
        if os.path.exists(path):
            print(f"Loading cached parquet from: {path}")
            return pd.read_parquet(path)
        return None

    def save_parquet(self, df, filename):
        """
        Saves a DataFrame to a parquet file, ensuring parent directories exist.
        """
        path = self._resolve_path(filename)
        print(f"Saving parquet to: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    def load_numpy(self, filename):
        """
        Attempts to load a numpy array from a .npy file.
        Returns None if the file does not exist.
        """
        path = self._resolve_path(filename)
        if os.path.exists(path):
            print(f"Loading cached numpy from: {path}")
            return np.load(path)
        return None

    def save_numpy(self, array, filename):
        """
        Saves a numpy array to a .npy file, ensuring parent directories exist.
        """
        path = self._resolve_path(filename)
        print(f"Saving numpy to: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)

    def exists(self, filename):
        """
        Checks if a specific file exists in the cache.
        """
        path = self._resolve_path(filename)
        return os.path.exists(path)
