import os
import random
import numpy as np
import torch
import hashlib
import json
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_log_loss(y_true, y_pred, labels=None):
    """
    Computes the multi-class logarithmic loss with specific normalization and clipping.

    Per task specifications:
    1. Probabilities are rescaled (each row divided by row sum).
    2. Probabilities are clipped to [1e-15, 1 - 1e-15].

    Args:
        y_true: Ground truth labels (array-like). Can be strings or encoded integers.
        y_pred: Predicted probabilities (array-like of shape [n_samples, n_classes]).
        labels: Optional list of labels to ensure correct column mapping if y_true are strings.

    Returns:
        float: The calculated log loss.
    """
    # Ensure y_pred is a numpy array
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential division by zero (though unlikely with valid model outputs)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip probabilities to avoid extremes of the log function
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Compute Log Loss
    # sklearn.metrics.log_loss handles the log calculation
    return log_loss(y_true, y_pred, labels=labels)


def generate_config_hash(config_dict):
    """
    Generates a unique MD5 hash for a configuration dictionary.
    Used for caching intermediate data processing artifacts.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: A hexadecimal hash string.
    """
    # Sort keys to ensure the hash is deterministic regardless of key insertion order
    # default=str handles non-serializable types by converting them to strings
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
