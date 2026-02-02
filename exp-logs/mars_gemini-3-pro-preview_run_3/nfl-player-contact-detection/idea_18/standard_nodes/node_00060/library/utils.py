import os
import random
import numpy as np
import torch
import hashlib
import json
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch libraries.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed for dictionary iteration consistency
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC) between true and predicted labels.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def get_hash(obj):
    """
    Generates a unique MD5 hash for a given object (e.g., dictionary, list, string).
    Used for creating deterministic cache keys based on feature configurations.

    Args:
        obj (any): The object to hash. Dictionaries are sorted by key to ensure consistency.

    Returns:
        str: The hexadecimal hash string.
    """
    if isinstance(obj, dict):
        # Serialize dictionary to JSON with sorted keys for deterministic hashing
        # default=str handles non-serializable types gracefully by converting to string
        obj_str = json.dumps(obj, sort_keys=True, default=str)
    else:
        obj_str = str(obj)

    # Create MD5 hash
    hash_object = hashlib.md5(obj_str.encode("utf-8"))
    return hash_object.hexdigest()
