import os
import random
import numpy as np
import torch
import json
import hashlib
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def get_data_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash for a given configuration dictionary.
    This is used to invalidate caches when configuration parameters change.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Sort keys to ensure consistent ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create MD5 hash
    hash_object = hashlib.md5(config_str.encode("utf-8"))
    return hash_object.hexdigest()
