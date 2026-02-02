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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC) between true labels and predictions.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def get_data_hash(config_data) -> str:
    """
    Generates a unique MD5 hash for a given configuration state.
    This is used to detect changes in parameters and invalidate data caches.

    Args:
        config_data (dict or object): The configuration data to hash.
                                      Can be a dictionary or a class with attributes.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Normalize input to a dictionary
    if not isinstance(config_data, dict):
        try:
            # Attempt to get attributes if it's a class/object
            data_to_hash = vars(config_data)
        except TypeError:
            # Fallback if vars() fails or it's a simple type
            data_to_hash = config_data
    else:
        data_to_hash = config_data

    # Serialize to JSON with sorted keys for determinism.
    # default=str ensures that non-serializable objects (like Paths or types) are converted to strings.
    serialized_config = json.dumps(data_to_hash, sort_keys=True, default=str)

    # Create MD5 hash
    hasher = hashlib.md5()
    hasher.update(serialized_config.encode("utf-8"))

    return hasher.hexdigest()
