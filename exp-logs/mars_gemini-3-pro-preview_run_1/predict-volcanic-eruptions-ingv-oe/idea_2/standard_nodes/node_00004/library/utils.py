import os
import random
import numpy as np
import torch
import json
import hashlib
from library.config import FEATURE_CONFIG, get_config_hash


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
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


def compute_config_hash(config_dict=None):
    """
    Generates a deterministic MD5 hash of a configuration dictionary.
    Used to version cached feature files.

    Args:
        config_dict (dict, optional): The configuration dictionary to hash.
                                      If None, uses the default FEATURE_CONFIG via get_config_hash().

    Returns:
        str: The MD5 hash of the JSON-serialized configuration.
    """
    if config_dict is None:
        # Use the function provided in library.config to avoid re-implementation
        return get_config_hash()

    # Fallback logic for custom dictionaries to ensure flexibility
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
