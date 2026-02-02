import os
import random
import json
import hashlib
import numpy as np
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_config_hash(config):
    """
    Generates a SHA256 hash for a configuration dictionary or list.
    Ensures deterministic hashing by sorting keys and handling common types.

    Args:
        config (dict or list): The configuration object to hash.

    Returns:
        str: The hexadecimal SHA256 hash string.
    """
    # Serialize config to JSON with sorted keys to ensure determinism
    # default=str handles types like np.int64 or Path objects
    config_str = json.dumps(config, sort_keys=True, default=str)

    # Generate SHA256 hash
    full_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    return full_hash


def compute_metric(y_true, y_pred, labels=None):
    """
    Computes the multi-class log loss metric with specific normalization and clipping.

    The metric follows the competition rules:
    1. Rescale rows to sum to 1.
    2. Clip probabilities to [1e-15, 1-1e-15].
    3. Compute Log Loss.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        labels (list, optional): List of class labels to index the columns of y_pred.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide each row by its sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle rows that might sum to 0 (though unlikely with valid model output)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip: Avoid extremes of the log function
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Compute Log Loss
    return log_loss(y_true, y_pred, labels=labels)
