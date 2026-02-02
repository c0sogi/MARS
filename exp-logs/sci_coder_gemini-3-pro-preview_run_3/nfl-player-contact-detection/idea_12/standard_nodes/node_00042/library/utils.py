import os
import random
import numpy as np
import torch
import hashlib
import json
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

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


def calc_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, num_steps: int = 100
) -> tuple:
    """
    Finds the optimal probability threshold that maximizes MCC via linear search.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities from the model.
        num_steps (int): Number of steps to search between 0 and 1.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    best_threshold = 0.5
    best_score = -1.0

    # Generate thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        score = calc_mcc(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def get_data_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    Used for cache invalidation when feature parameters change.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: A hexadecimal hash string.
    """
    # Use json dumps with sort_keys=True to ensure consistent ordering
    # default=str handles non-serializable types by converting them to strings
    encoded = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()
