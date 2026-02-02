import os
import random
import numpy as np
import torch
import hashlib
import json
from sklearn.metrics import mean_absolute_error


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)


def get_config_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash for a configuration dictionary.
    Useful for caching mechanisms to detect changes in parameters.

    Args:
        config_dict (dict): The dictionary containing configuration parameters.

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Use json.dumps with sort_keys=True to ensure consistent ordering.
    # default=str handles non-serializable types (like numpy ints/floats) gracefully.
    encoded = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()
