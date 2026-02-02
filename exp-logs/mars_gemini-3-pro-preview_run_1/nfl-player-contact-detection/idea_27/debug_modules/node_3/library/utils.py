import os
import random
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
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


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


class CacheManager:
    """
    Manages saving and loading of intermediate artifacts to the working directory.
    Supports Parquet for DataFrames, NPY for NumPy arrays, and Joblib for models.
    """

    def __init__(self, working_dir=None):
        self.working_dir = working_dir if working_dir else Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_path(self, filename):
        return os.path.join(self.working_dir, filename)

    def generate_key(self, params_dict):
        """
        Generates a unique MD5 hash for a given dictionary of parameters.
        Useful for parameter-aware caching.
        """
        # Sort keys to ensure consistent ordering
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()

    def exists(self, filename):
        """Checks if a file exists in the working directory."""
        return os.path.exists(self._get_path(filename))

    def save_parquet(self, df, filename):
        """Saves a pandas DataFrame to a parquet file."""
        path = self._get_path(filename)
        df.to_parquet(path, index=False)
        # print(f"Saved DataFrame to {path}")

    def load_parquet(self, filename):
        """Loads a pandas DataFrame from a parquet file."""
        path = self._get_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache miss: {path} does not exist.")
        return pd.read_parquet(path)

    def save_npy(self, array, filename):
        """Saves a numpy array to an .npy file."""
        path = self._get_path(filename)
        np.save(path, array)
        # print(f"Saved NumPy array to {path}")

    def load_npy(self, filename):
        """Loads a numpy array from an .npy file."""
        path = self._get_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache miss: {path} does not exist.")
        return np.load(path)

    def save_joblib(self, obj, filename):
        """Saves an object (e.g., model) using joblib."""
        path = self._get_path(filename)
        joblib.dump(obj, path)
        # print(f"Saved object to {path}")

    def load_joblib(self, filename):
        """Loads an object using joblib."""
        path = self._get_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache miss: {path} does not exist.")
        return joblib.load(path)

    def clear(self):
        """Clears the working directory (use with caution)."""
        for f in os.listdir(self.working_dir):
            os.remove(os.path.join(self.working_dir, f))
