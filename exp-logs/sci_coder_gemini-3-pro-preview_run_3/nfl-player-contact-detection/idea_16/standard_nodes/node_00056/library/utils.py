import os
import random
import numpy as np
import torch
import json
import hashlib
from sklearn.metrics import matthews_corrcoef


def setup_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba):
    """
    Finds the probability threshold that maximizes the MCC score.
    Performs a linear search over thresholds from 0.01 to 0.99.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities of the positive class.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    best_mcc = -1.0
    best_thresh = 0.5

    # Search space: 0.01 to 0.99 with step 0.01
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred = (y_pred_proba >= thresh).astype(int)

        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def get_config_hash(config_dict):
    """
    Generates a unique MD5 hash for a configuration dictionary.
    Used for cache invalidation.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Serialize dictionary to JSON string with sorted keys to ensure determinism.
    # default=str handles non-serializable objects (like numpy types) by converting to string.
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create MD5 hash
    hash_obj = hashlib.md5(config_str.encode("utf-8"))
    return hash_obj.hexdigest()
