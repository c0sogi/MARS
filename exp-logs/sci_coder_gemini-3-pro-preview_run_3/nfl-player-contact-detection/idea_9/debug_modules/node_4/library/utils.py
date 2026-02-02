import os
import random
import numpy as np
import torch
import hashlib
import json
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value.
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
    return matthews_corrcoef(y_true, y_pred)


def generate_config_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash for a configuration dictionary.
    Useful for creating cache filenames based on feature parameters.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Serialize the dictionary to a JSON string with sorted keys to ensure determinism
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
