import os
import random
import hashlib
import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between the true and predicted values.
    """
    return mean_absolute_error(y_true, y_pred)


class CacheManager:
    """
    Manages configuration hashing to ensure data consistency and prevent usage of stale cache.
    """

    def __init__(self):
        # Define keys from Config that affect data preprocessing/generation.
        # Changes to these parameters should trigger cache invalidation.
        self.relevant_keys = [
            "N_SENSORS",
            "SAMPLE_RATE",
            "SIGNAL_LENGTH",
            "GLOBAL_MAX_READING",
            "FREQ_BANDS",
            "N_BLOCKS",
            "N_FFT",
            "HOP_LENGTH",
            "IMG_SIZE",
        ]

    def _compute_hash(self):
        """
        Computes an MD5 hash of the current relevant configuration settings.
        """
        config_values = {}
        for key in self.relevant_keys:
            if hasattr(Config, key):
                val = getattr(Config, key)
                # Convert values to string to handle dicts/tuples consistently
                config_values[key] = str(val)

        # Serialize with sort_keys=True to ensure deterministic string representation
        serialized_config = json.dumps(config_values, sort_keys=True)
        return hashlib.md5(serialized_config.encode("utf-8")).hexdigest()

    def is_cache_valid(self, file_path):
        """
        Determines if the cached file at file_path is valid for the current configuration.

        Args:
            file_path (str): Path to the cached data file.

        Returns:
            bool: True if file exists and its hash metadata matches current config, False otherwise.
        """
        if not os.path.exists(file_path):
            return False

        hash_file_path = f"{file_path}.hash"
        if not os.path.exists(hash_file_path):
            return False

        try:
            with open(hash_file_path, "r") as f:
                saved_hash = f.read().strip()
            current_hash = self._compute_hash()
            return saved_hash == current_hash
        except Exception:
            return False

    def update_cache_metadata(self, file_path):
        """
        Updates the hash metadata for a newly generated cache file.

        Args:
            file_path (str): Path to the cached data file.
        """
        hash_file_path = f"{file_path}.hash"
        current_hash = self._compute_hash()

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(hash_file_path)), exist_ok=True)

        with open(hash_file_path, "w") as f:
            f.write(current_hash)
